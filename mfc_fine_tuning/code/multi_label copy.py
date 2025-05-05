import json
import torch
import argparse
import os
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback, TextStreamer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import jsonlines as jl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from multi_label_prompt import PROMPT_MULTI
from tqdm import tqdm

class MFCFineTuner:
    def __init__(self, model_name, output_dir, save_path, json_output_file, file_name, subset_size):
        self.model_name = model_name
        self.output_dir = output_dir
        self.save_path = save_path
        self.json_output_file = json_output_file
        self.file_name = file_name
        self.subset_size = subset_size

        self.max_seq_length = 5000
        self.dtype = None
        self.load_in_4bit = True
        self.system_instruction = PROMPT_MULTI
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                                ### Instruction:
                                {}

                                ### Input:
                                Text to analyze: {}

                                ### Response:
                                {}"""

        self.model = None
        self.tokenizer = None
        self.EOS_TOKEN = None

    def formatting_prompts_func(self, example):
        """Format prompts for training"""
        instruction = self.system_instruction
        input_text = example['sentence']
        output = example['label']
        prompt = self.alpaca_prompt.format(instruction, input_text, output) + self.EOS_TOKEN
        return {"text": prompt}
    
    def create_subset(self, dataset):
        if self.subset_size is None:
            return dataset 

        shuffled_dataset = dataset.shuffle(seed=42)
        subset = shuffled_dataset.select(range(self.subset_size))
        return subset

    def evaluate(self, eval_set):
        """Evaluate the model on the evaluation set"""
        FastLanguageModel.for_inference(self.model)
        true_labels = []
        predictions = []
        to_write = []
        
        for sample in tqdm(eval_set):
            sample_input = sample['text'].split('### Response:')[0] + '### Response:'
            sample_label = sample['text'].split('### Response:')[1].strip()
            #   sample_label = sample_label.strip('<|im_end|>')
            sample_label = sample_label.strip('<|eot_id|>')
            
            inputs = self.tokenizer(sample_input, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=10, use_cache=True)
            text_output = self.tokenizer.batch_decode(outputs)[0]
            print("RESPONSE:   ", text_output)
            prediction = text_output.split('### Response:')[1]
            prediction = prediction.strip()
            prediction = prediction.strip('<|eot_id|>')            # prediction = prediction.strip('<|im_end|>')
            prediction = prediction.split('\n', 1)[0]
            prediction = prediction.rstrip(',')
            print("EXTRACTED PRED:    ", prediction)
        #    prediction_labels = [int(label) for label in prediction.split(',') if label.strip().isdigit()]
        #    prediction_labels = list(set(prediction_labels))
        #    print(prediction_labels)
            to_write.append({"label": sample_label, "prediction": prediction})
            true_labels.append([label for label in sample_label])
            predictions.append(prediction)

     #   mlb = MultiLabelBinarizer()
     #   true_labels_bin = mlb.fit_transform(true_labels)
     #   predictions_bin = mlb.transform(predictions)
        
     #   accuracy = accuracy_score(true_labels_bin, predictions_bin)
     #   precision = precision_score(true_labels_bin, predictions_bin, average='macro')
     #   recall = recall_score(true_labels_bin, predictions_bin, average='macro')
     #   f1 = f1_score(true_labels_bin, predictions_bin, average='macro')
        
      #  print("\nEvaluation Metrics:")
       # print(f"Accuracy: {accuracy:.4f}")
       # print(f"Precision: {precision:.4f}")
       # print(f"Recall: {recall:.4f}")
       # print(f"F1-Score: {f1:.4f}")

        return to_write

    def main(self):
        """Main function to fine-tune the model"""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            cache_dir="/data/gpfs/projects/punim0478/guida/models",
            device_map="auto",
            trust_remote_code=True
        )

        self.EOS_TOKEN = self.tokenizer.eos_token

        dataset = load_dataset('json', data_files=self.file_name)['train']
        if self.subset_size:
            dataset = self.create_subset(dataset)
        
        dataset = dataset.map(self.formatting_prompts_func)

        print(f"\nTotal dataset size after subsetting: {len(dataset)}")
        
        train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
        print(train_testvalid)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
        
        train_dataset = train_testvalid['train']
        val_dataset = test_valid['train']
        test_dataset = test_valid['test']

        print(f"\nDataset splits:")
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        num_epochs = 8
        train_size = len(train_dataset)
        per_device_batch_size = 8
        gradient_accumulation_steps = 8
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps

        print(f"\nTraining Configuration:")
        print(f"Total training examples: {train_size}")
        print(f"Effective batch size: {effective_batch_size}")


        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            args=SFTConfig(
               # dataset_text_field="text",
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=0.05,
                num_train_epochs=num_epochs,
                learning_rate=1e-5,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=50,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                output_dir=self.output_dir,
                evaluation_strategy="epoch",  
                save_strategy="epoch",        
                save_total_limit=1, 
                max_seq_length=self.max_seq_length,
                dataset_num_proc=4,
                packing=True,
                )
        )

        trainer_stats = trainer.train()

        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

        to_write = self.evaluate(test_dataset)
        with jl.open(self.json_output_file, 'w') as w:
            for item in to_write:
                w.write(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on the MFC dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to fine-tune.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save training outputs.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument("--json_output_file", type=str, required=True, help="Path to save evaluation results.")
    parser.add_argument("--file_name", type=str, required=True, help="Path to the MFC dataset file.")
    parser.add_argument("--subset_size", type=int, default=None, help="Size of dataset subset to use (optional)")

    args = parser.parse_args()

    fine_tuner = MFCFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        save_path=args.save_path,
        json_output_file=args.json_output_file,
        file_name=args.file_name,
        subset_size=args.subset_size
    )
    fine_tuner.main()