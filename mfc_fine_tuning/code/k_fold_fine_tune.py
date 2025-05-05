import json
import torch
import argparse
import os
import re
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback, TextStreamer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import jsonlines as jl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from multi_class_prompt import PROMPT_MULTI
from tqdm import tqdm

def formatting_prompts_func(example, eos_token):
    instruction = PROMPT_MULTI
    input_text = example['sentence']
    output = example['label']
    
    prompt = alpaca_prompt.format(instruction, input_text, output) + eos_token
    return {"text": prompt}

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Article to analyze: {}

### Response:
{}"""

def evaluate(eval_set, model, tokenizer):
    FastLanguageModel.for_inference(model)
    to_write = []
    
    for sample in tqdm(eval_set):
        sample_input = sample['text'].split('### Response:')[0] + '### Response:'
        sample_label = sample['text'].split('### Response:')[1].strip()
        sample_label = sample_label.strip('<|im_end|>').strip('<|endoftext|>')
        
        inputs = tokenizer(sample_input, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
        text_output = tokenizer.batch_decode(outputs)[0]
        prediction = text_output.split('### Response:')[1].strip().split('\n')[0]
        
        to_write.append({"label": int(sample_label), "prediction": int(prediction)})
    
    return to_write

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--json_output_file", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True)
    args = parser.parse_args()
    
    max_seq_length = 2048
    dtype = None
    load_in_4bit = False
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        cache_dir="/data/gpfs/projects/punim0478/guida/models",
        device_map="auto",
        trust_remote_code=True
    )
    
    EOS_TOKEN = tokenizer.eos_token
    
    dataset = load_dataset('json', data_files=args.file_name)['train']
   # dataset = dataset.shuffle(seed=42).select(range(100))
    
    dataset = dataset.map(lambda x: formatting_prompts_func(x, EOS_TOKEN))
    
    topics = list(set(dataset["topic"]))
    
    for fold, test_topic in enumerate(topics):
        if fold > 0:
            torch.cuda.empty_cache()
            
            # Reinitialize model and tokenizer for this fold
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                cache_dir="/data/gpfs/projects/punim0478/guida/models",
                device_map="auto",
                trust_remote_code=True
            )
        
        test_dataset = dataset.filter(lambda x: x['topic'] == test_topic)
        train_dataset_all = dataset.filter(lambda x: x['topic'] != test_topic)
        train_eval_split = train_dataset_all.train_test_split(test_size=0.2, seed=42)
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_eval_split['train'],
            eval_dataset=train_eval_split['test'],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            args=SFTConfig(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                num_train_epochs=3,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=50,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                seed=3407,
                output_dir=f"{args.output_dir}/fold_{fold}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                dataset_num_proc=4,
                packing=True,
            )
        )
        
        trainer.train()
        model.save_pretrained(f"{args.save_path}_fold_{fold}")
        tokenizer.save_pretrained(f"{args.save_path}_fold_{fold}")
        
        out_text = evaluate(test_dataset, model, tokenizer)
        
        with jl.open(f"{args.json_output_file}_test_{test_topic}.jsonl", 'w') as w:
            for item in out_text:
                w.write(item)
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()