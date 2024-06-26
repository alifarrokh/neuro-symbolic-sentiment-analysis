import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from evaluate import load as load_metric
from sentiment_analysis_model import RobertaForSentimentAnalysis
from load_datasets import load_sts


"""
Results
Very Base Model     0.9433497536945813 (RoBERTa's Classification Head)
Base Model          0.9482758620689655 (New Classification Head)
HAN Model           0.9458128078817734 (New Classification Head + HAN)
"""


# Hyperparameters
train_conf = {
    'output_dir': 'exps/roberta-sts-base',
    'batch_size': 16,
    'gradient_accumulation_steps': 1,
    'epochs': 5,
    'learning_rate': 5e-5,
    'warmup_epochs': 1,
    'log_ratio': 0.25,
    'weight_decay': 0.01,
    'with_han': False
}

# Load the tokenizer and data collator
MODEL = 'FacebookAI/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the dataset
dataset, num_labels = load_sts()
dataset = dataset.map(lambda item: tokenizer(item['text'], truncation=True), remove_columns=['text'])

# Evaluation metrics
accuracy = load_metric('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Load the model
config = AutoConfig.from_pretrained('FacebookAI/roberta-base')
config.update({'with_han': train_conf['with_han'], 'num_labels': num_labels})
model = RobertaForSentimentAnalysis.from_pretrained(MODEL, config=config)

# Verify that experiment folder does not exist
assert not os.path.exists(train_conf['output_dir']), "Experiment folder already exists."

# Define the training args
steps_per_epoch = int(len(dataset['train']) / (train_conf['batch_size'] * train_conf['gradient_accumulation_steps']))
log_steps = int(steps_per_epoch * train_conf['log_ratio'])
training_args = TrainingArguments(
    # Saving
    output_dir=train_conf['output_dir'],
    save_strategy="steps",
    save_steps=log_steps,
    load_best_model_at_end=True,
    save_total_limit=1,

    # Logging
    logging_dir=os.path.join(train_conf['output_dir'], 'logs'),
    logging_strategy="steps",
    logging_steps=log_steps,

    # Training
    group_by_length=True,
    per_device_train_batch_size=train_conf['batch_size'],
    per_device_eval_batch_size=train_conf['batch_size'],
    gradient_accumulation_steps=train_conf['gradient_accumulation_steps'],
    num_train_epochs=train_conf['epochs'],
    gradient_checkpointing=True,
    learning_rate=train_conf['learning_rate'],
    warmup_steps=int(steps_per_epoch * train_conf['warmup_epochs']),
    weight_decay=train_conf['weight_decay'],

    # Evaluation
    eval_strategy="steps",
    eval_steps=log_steps,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

# Evaluation
eval_results = trainer.evaluate(dataset['test'])
print(f'Best Evaluation Accuracy: {eval_results["eval_accuracy"]}')