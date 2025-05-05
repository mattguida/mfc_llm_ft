import json
import torch
import argparse
import os
import gc
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

class CrossTopicFineTuner:
    def __init__(self, model_name, output_dir, train_file):
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_file = train_file
        self.topics = ["immigration", "deathpenalty", "samesex", "guncontrol", "tobacco", "comments"]

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

    def load_and_filter_data(self, test_topic):
        full_dataset = load_dataset('json', data_files=self.train_file)['train']
        train_dataset = full_dataset.filter(lambda example: example['topic'] != test_topic)
        test_dataset = full_dataset.filter(lambda example: example['topic'] == test_topic)
        return train_dataset, test_dataset

    def evaluate(self, eval_set):
        """Evaluate the model on the evaluation set"""
        FastLanguageModel.for_inference(self.model)
        true_labels = []
        predictions = []
        to_write = []

        for sample in tqdm(eval_set):
            sample_input = sample['text'].split('### Response:')[0] + '### Response:'
            sample_label = sample['text'].split('### Response:')[1].strip()
            sample_label = sample_label.strip('<|eot_id|>')

            inputs = self.tokenizer(sample_input, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=10, use_cache=True)
            text_output = self.tokenizer.batch_decode(outputs)[0]
            
            # Clear GPU memory used for this sample
            del inputs, outputs
            torch.cuda.empty_cache()
            
          # print("RESPONSE:   ", text_output)
            prediction = text_output.split('### Response:')[1]
            prediction = prediction.strip()
            prediction = prediction.strip('<|eot_id|>')
            prediction = prediction.split('\n', 1)[0]
            print("EXTRACTED PRED:    ", prediction)s
            true_label = int(sample_label)
            to_write.append({"label": sample_label, "prediction": prediction})
            true_labels.append(true_label)
            predictions.append(prediction)

        return to_write

    def train_and_evaluate(self, test_topic):
        # Load and format data
        train_dataset, test_dataset = self.load_and_filter_data(test_topic)
        train_dataset = train_dataset.map(self.formatting_prompts_func)
        test_dataset = test_dataset.map(self.formatting_prompts_func)

        print(f"\nTotal training dataset size: {len(train_dataset)}")

        train_val = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_val['train']
        val_dataset = train_val['test']

        print(f"\nDataset splits:")
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        # Training setup
        num_epochs = 6
        train_size = len(train_dataset)
        per_device_batch_size = 4
        gradient_accumulation_steps = 4
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps

        print(f"\nTraining Configuration:")
        print(f"Total training examples: {train_size}")
        print(f"Effective batch size: {effective_batch_size}")

        # Get a fresh PEFT model for this topic
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
                output_dir=os.path.join(self.output_dir, f"outputs_{test_topic}"),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                max_seq_length=self.max_seq_length,
                dataset_num_proc=4,
                packing=True,
            )
        )

        trainer_stats = trainer.train()
        
        # Free up memory from trainer objects
        del trainer_stats
        
        # Run evaluation
        results = self.evaluate(test_dataset)
        
        # Save model and results
        save_path = os.path.join(self.output_dir, f"model_{test_topic}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        json_output_file = os.path.join(self.output_dir, f"results_{test_topic}.jsonl")
        with jl.open(json_output_file, 'w') as w:
            for item in results:
                w.write(item)
        
        # Clear memory from this topic's model
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Reload base model for next topic
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            cache_dir="/data/gpfs/projects/punim0478/guida/models",
            device_map="auto",
            trust_remote_code=True
        )

    def main(self):
        # Load initial model and tokenizer
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

        for topic in self.topics:
            print(f"Training and evaluating for test topic: {topic}")
            self.train_and_evaluate(topic)
            
            # Additional aggressive memory cleanup after each topic
            gc.collect()
            torch.cuda.empty_cache()
            
        # Final cleanup
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on the MFC dataset for cross-topic evaluation.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to fine-tune.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save training outputs and results.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the full dataset file.")

    args = parser.parse_args()

    fine_tuner = CrossTopicFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_file=args.train_file
    )
    fine_tuner.main()