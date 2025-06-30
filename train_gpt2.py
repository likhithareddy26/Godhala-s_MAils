import json
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load data
def load_data(path="final_data.jsonl"): 
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    return Dataset.from_list([{"text": item["prompt"] + item["completion"]} for item in lines])

# Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load dataset
dataset = load_data()

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Define Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-email",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    fp16=False  # Set True if using GPU with mixed precision
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save model
trainer.save_model("./gpt2-finetuned-email")
tokenizer.save_pretrained("./gpt2-finetuned-email")
