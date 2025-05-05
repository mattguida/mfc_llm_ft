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
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for multi-class classification")
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

def prepare_datasets(data, tokenizer, max_length, seed=42):
    # Extract sentences and labels
    sentences = [item['sentence'] for item in data]
    labels = [int(item['label']) for item in data]
    
    # Get unique labels and create label mapping
    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    mapped_labels = [label_map[label] for label in labels]
    
    # Create features dict for datasets library
    features = {
        'sentence': sentences,
        'label': mapped_labels,
        'topic': [item.get('topic', '') for item in data]  # Include topic if available
    }
    
    # Create a single dataset and split
    dataset = Dataset.from_dict(features)
    
    # Split into train, validation, test (80%, 10%, 10%)
    splits = dataset.train_test_split(test_size=0.2, seed=seed)
    train_dataset = splits['train']
    test_val = splits['test'].train_test_split(test_size=0.5, seed=seed)
    val_dataset = test_val['train']
    test_dataset = test_val['test']
    
    # Tokenize datasets with static padding
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
    
    # Load and prepare data
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} examples")
    
    if args.small_dataset:
        print("Using small dataset (100 instances) for testing")
        data = data[:100]
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, num_labels, label_map = prepare_datasets(
        data, tokenizer, args.max_length, args.seed
    )
    
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Number of classes: {num_labels}")
    print(f"Label mapping: {label_map}")
    
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        #report_to="wandb",
        save_total_limit=2,  # Keep only the 2 best checkpoints
        fp16=torch.cuda.is_available(),  # Use mixed precision training if available
        logging_strategy="epoch",
        warmup_ratio=0.1,  # 5% of training steps for warmup
        lr_scheduler_type="linear"  # Linear decrease after warmup
    )
    
    # Initialize trainer (without data_collator)
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
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")
    
    # Save the final model
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # Save label mapping
    with open(f"{args.output_dir}/final_model/label_map.json", 'w') as f:
        json.dump(label_map, f)
    
    # Generate detailed classification report
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    # Convert back to original labels for the report
    reverse_map = {v: k for k, v in label_map.items()}
    original_preds = [reverse_map[p] for p in preds]
    original_labels = [reverse_map[l] for l in labels]
    
    report = classification_report(original_labels, original_preds, digits=4)
    print("Classification Report:")
    print(report)
    
    with open(f"{args.output_dir}/classification_report.txt", 'w') as f:
        f.write(report)

    
    print(f"Training complete - Model saved to {args.output_dir}/final_model")

if __name__ == "__main__":
    main()