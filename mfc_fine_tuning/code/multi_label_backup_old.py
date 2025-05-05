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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from multi_label_prompt import PROMPT_MULTI
from collections import Counter


class MFCFineTuner:
    def __init__(self, model_name, output_dir, save_path, json_output_file, file_paths):
        self.model_name = model_name
        self.output_dir = output_dir
        self.save_path = save_path
        self.json_output_file = json_output_file
        self.file_paths = file_paths

        self.max_seq_length = 4000
        self.dtype = None
        self.load_in_4bit = True
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

    def load_mfc_data(self):
        """Load and preprocess MFC dataset files with error handling."""
        all_data = []
        
        for file_path in self.file_paths: 
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)

                for i, article in data.items():
                    if article['irrelevant'] != 1:
                        annotators = article['annotations']['framing']
                        num_annotators = len(annotators)
                        
                        unique_codes_per_annotator = []  
                        for annotator, annotations in annotators.items():
                            annotator_codes = set()  
                            for annotation in annotations:
                                if str(annotation['code']).endswith('0'):
                                    annotator_codes.add(annotation['code'])
                            unique_codes_per_annotator.append(annotator_codes)
                        
                        all_codes = [code for annotator_set in unique_codes_per_annotator for code in annotator_set]
                        code_counts = Counter(all_codes)
                        
                        agreed_codes = []
                        for code, count in code_counts.items():
                            if num_annotators == 2 and count == 2:
                                agreed_codes.append(int(float(code)))
                            elif num_annotators > 2 and count >= num_annotators / 2:
                                agreed_codes.append(int(float(code)))
                        
                        if agreed_codes:
                            all_data.append({
                                'unique_value': i,
                                'text': article['text'],
                                'labels': agreed_codes 
                            })
        
        return all_data  


    def formatting_prompts_func(self, example):
        """Format prompts for training."""
        instruction = self.system_instruction
        input_text = example['text']
        output = ", ".join(map(str, example['labels']))

        prompt = self.alpaca_prompt.format(instruction, input_text, output) + self.EOS_TOKEN
        return {"text": prompt}

    def evaluate(self, eval_set):
        """Evaluate the model on the evaluation set."""
        rows = []
        FastLanguageModel.for_inference(self.model)
        true_labels = []
        predictions = []
        to_write = []
        
        for sample in tqdm(eval_set):
            sample_input = sample['text'].split('### Response:')[0] + '### Response:'
            sample_label = sample['text'].split('### Response:')[1].strip()
            sample_label = sample_label.strip('<|im_start|>')
            sample_label = sample_label.strip('<|im_end|>')

            inputs = self.tokenizer(sample_input, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=10, use_cache=True)
            text_output = self.tokenizer.batch_decode(outputs)[0]
            prediction = text_output.split('### Response:')[1]
            prediction = prediction.strip()
            prediction = prediction.strip('<|im_start|>')
            prediction = prediction.strip('<|im_end|>')

          #  true_label_list = list(map(int, sample_label.split(',')))
          #  predicted_label_list = list(map(int, prediction.split(',')))

          #  true_labels.append(true_label_list)
          #  predictions.append(predicted_label_list)

            to_write.append({"label": sample_label, "prediction": prediction})

        return to_write

    def main(self):
        """Main function to fine-tune the model."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            cache_dir="/data/gpfs/projects/punim0478/guida/models"
        )

        self.EOS_TOKEN = self.tokenizer.eos_token
        

        data = self.load_mfc_data()
        dataset = Dataset.from_list(data)
        dataset = dataset.map(self.formatting_prompts_func) 

        train_indices, temp_indices = train_test_split(
            range(len(dataset)),
            test_size=0.2,  # 20% for validation + test
            random_state=42
        )

        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,  # 10% for validation, 10% for test
            random_state=42
        )

        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)
        test_dataset = dataset.select(test_indices)
        
        num_epochs = 7
        train_size = len(train_dataset)
        per_device_batch_size = 8
        gradient_accumulation_steps = 4
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps
        steps_per_epoch = train_size // effective_batch_size
        total_steps = steps_per_epoch * num_epochs

        eval_steps = steps_per_epoch // 6
        save_steps = eval_steps * 4

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
            dataset_num_proc=2,
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

        to_write = self.evaluate(test_dataset)

        with jl.open(self.json_output_file, 'a') as w:
            for item in to_write:
                w.write(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on the MFC dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to fine-tune.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save training outputs.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument("--json_output_file", type=str, required=True, help="Path to save evaluation results.")
    parser.add_argument("--file_paths", nargs='+', required=True, help="Paths to the MFC dataset files.")

    args = parser.parse_args()

    fine_tuner = MFCFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        save_path=args.save_path,
        json_output_file=args.json_output_file,
        file_paths=args.file_paths
    )
    fine_tuner.main()