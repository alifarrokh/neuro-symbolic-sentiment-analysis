import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import DatasetDict
from lexical_substitution_model import (
    LexicalSubstitutionInputFormatter,
    LexicalSubstitutionDataCollator,
    RobertaForLexicalSubstitution,
)
from load_datasets import load_lexical_substitution_dataset


# Create cache directory
CACHE_DIR = 'cache'
os.system(f'mkdir -p {CACHE_DIR}')


# Hyperparameters
train_conf = {
    'output_dir': 'exps/roberta-ls',
    'batch_size': 32,
    'gradient_accumulation_steps': 1,
    'epochs': 20,
    'learning_rate': 1e-4,
    'warmup_epochs': 2,
    'log_ratio': 0.25,
    'weight_decay': 0.01,
}

# Load the tokenizer and data collator
MODEL = 'FacebookAI/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
input_formatter = LexicalSubstitutionInputFormatter(tokenizer)
data_collator = LexicalSubstitutionDataCollator(tokenizer)

# Load and prepare the dataset
dataset = load_lexical_substitution_dataset()
dataset_train = dataset['train'].map(input_formatter, remove_columns=dataset['train'].column_names, cache_file_name=f'{CACHE_DIR}/ls_train')
dataset_test = dataset['test'].map(input_formatter, remove_columns=dataset['train'].column_names, cache_file_name=f'{CACHE_DIR}/ls_test')
dataset = DatasetDict(train=dataset_train, test=dataset_test)


# Evaluation metrics
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    accuracy = (preds == labels).sum() / len(preds)
    return {'accuracy': accuracy}


# Load the model
model = RobertaForLexicalSubstitution.from_pretrained(MODEL)

# Verify that experiment folder does not exist
assert not os.path.exists(train_conf['output_dir']), "Experiment folder already exists."

# Define the training args
steps_per_epoch = int(len(dataset['train']) / (train_conf['batch_size'] * train_conf['gradient_accumulation_steps']))
log_steps = max(int(steps_per_epoch * train_conf['log_ratio']), 1)
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
    per_device_train_batch_size=train_conf['batch_size'],
    per_device_eval_batch_size=train_conf['batch_size'],
    gradient_accumulation_steps=train_conf['gradient_accumulation_steps'],
    num_train_epochs=train_conf['epochs'],
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
