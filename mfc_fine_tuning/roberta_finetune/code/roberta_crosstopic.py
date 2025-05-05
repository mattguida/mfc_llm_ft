import json
import torch
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaForSequenceClassification, 
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for multi-class classification with leave-one-out topic validation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON data file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--small_dataset", action="store_true", help="Use only 100 instances for testing")
    return parser.parse_args()

def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        # Try reading as JSONL if JSON parsing fails
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data

def prepare_datasets(train_data, val_data, test_data, tokenizer, max_length, seed=42):
    # Extract data for each split
    def extract_features(data):
        return {
            'sentence': [item['sentence'] for item in data],
            'label': [int(item['label']) for item in data],
            'topic': [item.get('topic', '') for item in data]
        }
    
    train_features = extract_features(train_data)
    val_features = extract_features(val_data)
    test_features = extract_features(test_data)
    
    # Get unique labels and create label mapping from all data
    all_labels = train_features['label'] + val_features['label'] + test_features['label']
    unique_labels = sorted(list(set(all_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Apply label mapping
    train_features['label'] = [label_map[label] for label in train_features['label']]
    val_features['label'] = [label_map[label] for label in val_features['label']]
    test_features['label'] = [label_map[label] for label in test_features['label']]
    
    # Create datasets
    train_dataset = Dataset.from_dict(train_features)
    val_dataset = Dataset.from_dict(val_features)
    test_dataset = Dataset.from_dict(test_features)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'], 
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Keep useful columns and format for Trainer
    tokenized_train = tokenized_train.remove_columns(['sentence', 'topic'])
    tokenized_val = tokenized_val.remove_columns(['sentence', 'topic'])
    tokenized_test = tokenized_test.remove_columns(['sentence', 'topic'])
    
    tokenized_train.set_format('torch')
    tokenized_val.set_format('torch')
    tokenized_test.set_format('torch')
    
    return tokenized_train, tokenized_val, tokenized_test, len(unique_labels), label_map

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} examples")
    
    if args.small_dataset:
        print("Using small dataset (100 instances) for testing")
        data = data[:100]
    
    # Get unique topics
    topics = sorted(list(set(item['topic'] for item in data if 'topic' in item)))
    print(f"Found {len(topics)} unique topics: {topics}")
    
    # Save all results for final comparison
    all_results = {}
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    # Perform leave-one-out cross-validation
    for held_out_topic in topics:
        print(f"\n{'='*50}")
        print(f"LEAVE-ONE-OUT VALIDATION: HELD-OUT TOPIC = '{held_out_topic}'")
        print(f"{'='*50}")
        
        # Split data
        train_val_data = [item for item in data if item.get('topic') != held_out_topic]
        test_data = [item for item in data if item.get('topic') == held_out_topic]
        
        # Split training data into train and validation
        train_data, val_data = train_test_split(
            train_val_data, test_size=0.1, random_state=args.seed
        )
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset, num_labels, label_map = prepare_datasets(
            train_data, val_data, test_data, tokenizer, args.max_length, args.seed
        )
        
        print(f"Train: {len(train_dataset)} examples")
        print(f"Validation: {len(val_dataset)} examples")
        print(f"Test (held-out topic '{held_out_topic}'): {len(test_dataset)} examples")
        print(f"Number of classes: {num_labels}")
        
        # Initialize model
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=num_labels
        )
        
        # Define training arguments
        topic_output_dir = os.path.join(args.output_dir, f"topic_{held_out_topic}")
        os.makedirs(topic_output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=topic_output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            logging_strategy="epoch"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Evaluate on validation set (in-domain topics)
        val_results = trainer.evaluate()
        print(f"Validation results (in-domain topics): {val_results}")
        
        # Evaluate on held-out topic
        print(f"Evaluating on held-out topic '{held_out_topic}'...")
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results (held-out topic): {test_results}")
        
        # Generate detailed classification report
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids
        
        # Convert back to original labels for the report
        reverse_map = {v: k for k, v in label_map.items()}
        original_preds = [reverse_map[p] for p in preds]
        original_labels = [reverse_map[l] for l in labels]
        
        # Generate classification report
        report = classification_report(original_labels, original_preds, digits=4)
        print(f"Classification Report for held-out topic '{held_out_topic}':")
        print(report)
        
        # Save results
        with open(os.path.join(topic_output_dir, "classification_report.txt"), 'w') as f:
            f.write(f"Classification Report for held-out topic '{held_out_topic}':\n")
            f.write(report)
        
        # Save model and tokenizer
        model_save_path = os.path.join(topic_output_dir, "final_model")
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        # Save label mapping
        with open(os.path.join(model_save_path, "label_map.json"), 'w') as f:
            json.dump(label_map, f)
        
        # Store results for comparison
        all_results[held_out_topic] = {
            "validation": val_results,
            "test": test_results
        }
    
    # Save overall results summary
    print("\n\n=== OVERALL RESULTS SUMMARY ===")
    for topic, results in all_results.items():
        print(f"\nTopic: {topic}")
        print(f"  In-domain validation F1: {results['validation']['eval_f1']:.4f}")
        print(f"  Held-out topic test F1: {results['test']['eval_f1']:.4f}")
        print(f"  Performance gap: {results['validation']['eval_f1'] - results['test']['eval_f1']:.4f}")
    
    # Save summary to file
    with open(os.path.join(args.output_dir, "leave_one_out_summary.txt"), 'w') as f:
        f.write("=== LEAVE-ONE-OUT CROSS-VALIDATION SUMMARY ===\n\n")
        for topic, results in all_results.items():
            f.write(f"Topic: {topic}\n")
            f.write(f"  In-domain validation F1: {results['validation']['eval_f1']:.4f}\n")
            f.write(f"  Held-out topic test F1: {results['test']['eval_f1']:.4f}\n")
            f.write(f"  Performance gap: {results['validation']['eval_f1'] - results['test']['eval_f1']:.4f}\n\n")
    
    print("\nLeave-one-out cross-validation complete!")
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()