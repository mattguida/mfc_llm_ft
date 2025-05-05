import json
import torch
import argparse
import os
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import jsonlines as jl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from multi_label_prompt import PROMPT_MULTI

class MFCFineTuner:
    def __init__(self, model_name, output_dir, save_path, json_output_file, train_file, test_file):
        self.model_name = model_name
        self.output_dir = output_dir
        self.save_path = save_path
        self.json_output_file = json_output_file
        self.train_file = train_file
        self.test_file = test_file

        self.max_seq_length = 4000
        self.dtype = None
        self.load_in_4bit = False
        self.system_instruction = PROMPT_MULTI
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                                ### Instruction:
                                {}

                                ### Input:
                                Article to analyze: {}

                                ### Response:
                                {}"""

        self.model = None
        self.tokenizer = None
        self.EOS_TOKEN = None

    def load_mfc_data(self, file_path, is_test=False):
        """Load and preprocess dataset file"""
        all_data = []
        
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if is_test:
                    all_data.extend([{
                        'unique_id': item['unique_id'],
                        'text': item['text']
                    } for item in data])
                else:
                    all_data.extend(data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        print(f"Total processed articles: {len(all_data)}")
        return all_data

    def formatting_prompts_func(self, example):
        """Format prompts for training"""
        instruction = self.system_instruction
        input_text = example['text']
        
        if 'labels' in example:
            output = ", ".join(map(str, example['labels']))
            prompt = self.alpaca_prompt.format(instruction, input_text, output) + self.EOS_TOKEN
        else:
            prompt = self.alpaca_prompt.format(instruction, input_text, "")
            
        return {"text": prompt}

    def evaluate_test(self, test_dataset):
        """Evaluate the model on the test set"""
        FastLanguageModel.for_inference(self.model)
        predictions = []
        
        for sample in tqdm(eval_set):
            sample_input = sample['text'].split('### Response:')[0] + '### Response:'
            sample_label = sample['text'].split('### Response:')[1].strip()
            sample_label = sample_label.strip('<|im_end|>')
            
            inputs = self.tokenizer(sample_input, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=10, use_cache=True)
            text_output = self.tokenizer.batch_decode(outputs)[0]
            
            prediction = text_output.split('### Response:')[1]
            prediction = prediction.strip()
            prediction = prediction.strip('<|im_end|>')
            
            prediction_labels = [int(label) for label in prediction.split(',') if label.strip().isdigit()]
            prediction_labels = list(set(prediction_labels))
            
            predictions.append({
                "unique_id": sample['unique_id'],
                "predicted_labels": prediction_labels
            })
        
        with open(self.json_output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"\nPredictions saved to {self.json_output_file}")
        return predictions

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

        train_data = self.load_mfc_data(self.train_file)
        dataset = Dataset.from_list(train_data)
        dataset = dataset.map(self.formatting_prompts_func)

        train_dataset, val_dataset = train_test_split(
            dataset,
            test_size=0.2,
            random_state=42
        )
        train_dataset = Dataset.from_list(train_dataset)
        val_dataset = Dataset.from_list(val_dataset)

        num_epochs = 1
        train_size = len(train_dataset)
        per_device_batch_size = 2
        gradient_accumulation_steps = 1
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps
        steps_per_epoch = train_size // effective_batch_size
        total_steps = steps_per_epoch * num_epochs

        eval_steps = steps_per_epoch // 2
        save_steps = eval_steps * 2

        print(f"\nTraining Configuration:")
        print(f"Total training examples: {train_size}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")

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
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=4,
            packing=True,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            args=TrainingArguments(
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=0.1,
                num_train_epochs=num_epochs,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=50,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                load_best_model_at_end=True,
                output_dir=self.output_dir,
                evaluation_strategy="steps",
                eval_steps=eval_steps,
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=3
            )
        )

        trainer_stats = trainer.train()

        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

        test_data = self.load_mfc_data(self.test_file, is_test=True)
        test_dataset = Dataset.from_list(test_data)
        predictions = self.evaluate_test(test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on the MFC dataset and test on separate data.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to fine-tune.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save training outputs.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument("--json_output_file", type=str, required=True, help="Path to save test predictions.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the MFC training dataset file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset file.")

    args = parser.parse_args()

    fine_tuner = MFCFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        save_path=args.save_path,
        json_output_file=args.json_output_file,
        train_file=args.train_file,
        test_file=args.test_file
    )
    fine_tuner.main()