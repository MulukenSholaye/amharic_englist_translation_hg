import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
from tqdm import tqdm
import os

# Configuration
MODEL_NAME = "facebook/nllb-200-3.3B"  # Use "facebook/nllb-200-distilled-600M" for smaller GPU memory
SRC_LANG = "amh"  # Amharic language code
TGT_LANG = "eng"  # English language code
MAX_LENGTH = 128
BATCH_SIZE = 4  # Reduce if OOM errors occur
GRAD_ACCUM_STEPS = 8  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
OUTPUT_DIR = "nllb-amh-eng-bible"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load Bible dataset (OPUS Books)
def load_bible_data():
    dataset = load_dataset("opus_books", "en-am")["train"]
    
    # Filter for Bible verses only
    bible_data = dataset.filter(lambda x: x["meta"]["source"] == "Bible")
    
    # Prepare samples
    samples = []
    for item in bible_data:
        samples.append({
            "amharic": item["translation"]["am"],
            "english": item["translation"]["en"]
        })
    
    # Split into train/validation
    split = int(0.9 * len(samples))
    train_data = samples[:split]
    val_data = samples[split:]
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

train_dataset, val_dataset = load_bible_data()

# Tokenization function
def preprocess_function(examples):
    inputs = [ex for ex in examples["amharic"]]
    targets = [ex for ex in examples["english"]]
    
    model_inputs = tokenizer(
        inputs, 
        text_target=targets,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # For NLLB, set the forced_bos_token_id for target language
    model_inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id[TGT_LANG]
    return model_inputs

# Tokenize datasets
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

# Metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute metrics
    bleu_result = bleu.compute(
        predictions=decoded_preds, 
        references=[[ref] for ref in decoded_labels]
    )
    meteor_result = meteor.compute(
        predictions=decoded_preds, 
        references=decoded_labels
    )
    
    return {
        "bleu": bleu_result["bleu"],
        "meteor": meteor_result["meteor"]
    }

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    fp16=True if device == "cuda" else False,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    report_to="tensorboard",
    save_strategy="epoch