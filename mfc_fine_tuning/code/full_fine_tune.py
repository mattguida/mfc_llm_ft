import json
import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import jsonlines as jl
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

def load_mfc_data(file_paths):
    """Load and preprocess MFC dataset files with error handling."""
    all_data = []
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in {file_path}")
                    print(f"Error occurs at position {e.pos}")
                    print(f"Line number: {e.lineno}, Column: {e.colno}")
                    continue
                
                for i, article in data.items():
                    try:
                        if article.get('primary_frame') is not None:
                            frame = str(article['primary_frame'])
                            all_data.append({
                                    'text': article['text'],
                                    'label': int(float(frame)),
                                    'topic': file_path.split('/')[-2]
                                })
                    except KeyError as e:
                        print(f"Missing key {e} in article {i}")
                    except Exception as e:
                        print(f"Error processing article {i}: {e}")
                        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    print(f"Total processed articles: {len(all_data)}")
    return all_data

file_paths = [
    '/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/guncontrol_labeled.json',
    '/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/immigration_labeled.json',
    '/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/samesex_labeled.json',
    '/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/tobacco_labeled.json',
    '/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/deathpenalty_labeled.json'
]

system_instruction = '''You are analyzing how news articles are framed in their coverage of controversial topics (gun control, immigration, gun control, death penalty, smoking and same-sex marriage). 
Your task is to identify the primary frame - the dominant way the article structures its narrative and presents the issue. 

Instructions:
1. Read the news article carefully
2. Identify the SINGLE most dominant frame that shapes the article's overall narrative
3. Output only the numerical code (i.e: 10, 7, etc.) for that primary frame

- Provide only the numerical code as your response
- Choose only one primary frame code
- Focus on the overall emphasis of the article, not individual mentions

Classify the primary frame choosing one numerical label from the following list:

    1 - Economic: Article primarily focuses on costs, benefits, or other financial implications.,
    
    2 - Capacity and resources: Article primarily discusses the availability of physical, human, or financial resources, and the capacity of current systems.,
    
    3 - Morality: Article primarily addresses religious or ethical implications.,
    
    4 - Fairness and equality: Article primarily focuses on the balance or distribution of rights, responsibilities, and resources.,
    
    5 - Legality, constitutionality and jurisprudence: Article primarily deals with rights, freedoms, and authority of individuals, corporations, and government.,
    
    6 - Policy prescription and evaluation: Article primarily discusses specific policies aimed at addressing problems.,
    
    7 - Crime and punishment: Article primarily focuses on the effectiveness and implications of laws and their enforcement.,
    
    8 - Security and defense: Article primarily deals with threats to the welfare of the individual, community, or nation.,
    
    9 - Health and safety: Article primarily focuses on healthcare, sanitation, and public safety.,
    
    10 - Quality of life: Article primarily addresses threats and opportunities for the individual’s wealth, happiness, and well-being.,
    
    11 - Cultural identity: Article primarily discusses traditions, customs, or values of a social group in relation to a policy issue.,
    
    12 - Public opinion: Article primarily focuses on attitudes and opinions of the general public, including polling and demographics.,
    
    13 - Political: Article primarily focuses on considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters.,
    
    14 - External regulation and reputation: Article primarily deals with international reputation or foreign policy of the U.S.,
    
    15 - Other: Article primarily covers any coherent group of frames not covered by the above categories.
'''

max_seq_length = 4000
dtype = None
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(example):
    instruction = system_instruction
    input_text = example['text']
    output = example['label']
    
    prompt = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
    
    return {"text": prompt}

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Article to analyze: {}

### Response:
{}"""



def evaluate(eval_set, model, tokenizer):
    rows = []
    FastLanguageModel.for_inference(model)
    true_labels = []
    predictions = []
    to_write = []
    for sample in tqdm(eval_set):
        sample_input = sample['text'].split('### Response:')[0] + '### Response:'
        sample_label = sample['text'].split('### Response:')[1].strip()
        sample_label = sample_label.strip('<|end_of_text|>')
        sample_label = sample_label.strip('<|eot_i')

        inputs = tokenizer(sample_input, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
        text_output = tokenizer.batch_decode(outputs)[0]
        prediction = text_output.split('### Response:')[1]
        prediction = prediction.strip()
        prediction = prediction.strip('<|end_of_text|>')
        prediction = prediction.strip('<|eot_id|>')

        true_labels.append(sample_label)
        predictions.append(prediction)
        
        to_write.append({"label": sample_label, "prediction": prediction})
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    return accuracy, precision, recall, f1, to_write

data = load_mfc_data(file_paths)
dataset = Dataset.from_list(data)
dataset = dataset.map(formatting_prompts_func)


train_indices, temp_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=dataset['label'],
    random_state=42
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.5,
    stratify=[dataset[i]['label'] for i in temp_indices],
    random_state=42
)

train_dataset = dataset.select(train_indices)
val_dataset = dataset.select(val_indices)
test_dataset = dataset.select(test_indices)

def print_label_distribution(dataset, name):
    label_counts = {}
    for item in dataset:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"\nLabel distribution in {name} set:")
    for label in sorted(label_counts.keys()):
        print(f"Label {label}: {label_counts[label]} ({label_counts[label]/len(dataset)*100:.2f}%)")

print_label_distribution(train_dataset, "train")
print_label_distribution(val_dataset, "validation")
print_label_distribution(test_dataset, "test")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    cache_dir="/data/gpfs/projects/punim0478/guida/models"
)

model = FastLanguageModel.get_peft_model(
    model,
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
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  
    dataset_text_field="text",
    max_seq_length=max_seq_length,
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
        output_dir="outputs_full_frame_classification_single_model_ds-qwen_7e_8x4",
        evaluation_strategy="steps",
        eval_steps=eval_steps,  
        save_strategy="steps",
        save_steps=save_steps, 
        save_total_limit=3
    )
)

trainer_stats = trainer.train()

save_path = 'train_ds-distill-qw15b_lora_fullframes_final'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
out_text = evaluate(test_dataset, model, tokenizer)

accuracy, precision, recall, f1, to_write = out_text

with jl.open('output_ds-distill-qwen_fullframes_mfc_articles_7e8x4.jsonl', 'w') as w:
    for item in to_write:
        w.write(item)

print(f"Final Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")