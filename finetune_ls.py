import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from transformers import (
    AutoTokenizer,
)
from datasets import DatasetDict
from load_datasets import load_lexical_substitution_dataset


# Load the tokenizer and data collator
MODEL = 'FacebookAI/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# Prepare the input
def prepare_input(sample):
    bos = tokenizer.get_vocab()[tokenizer.bos_token]
    sep = tokenizer.get_vocab()[tokenizer.sep_token]
    eos = tokenizer.get_vocab()[tokenizer.eos_token]

    sentence = sample['sentence']
    candidates = sample['candidates']
    label = sample['label']

    # Tokenize the sentencce
    left, target_token, right = re.split('<head>|</head>', sentence)
    if len(left) > 0 and left[-1] == ' ':
        left = left[:-1]
        target_token = ' ' + target_token
    tokenized_parts = tokenizer([left, target_token, right], add_special_tokens=False)['input_ids']

    # Tokenize the candidates
    label_index = candidates.index(label)
    tokenized_candidates = tokenizer(candidates, add_special_tokens=False)['input_ids']
    tokenized_candidates_lengths = [len(c)+1 for c in tokenized_candidates]

    # Concatenate the ids
    sentence_ids = [bos] + [t for part in tokenized_parts for t in part] + [eos]
    candidate_ids = [t for candidate in tokenized_candidates for t in (candidate + [sep])]

    # Verify the manually-created sequence
    assert tokenizer(re.sub('<head>|</head>', '', sentence))['input_ids'] == sentence_ids

    # A helper function to get the index of the last token in each candidate
    candidate_last_token_index = lambda c_index: len(sentence_ids) + sum(tokenized_candidates_lengths[:c_index]) + tokenized_candidates_lengths[c_index] - 2

    item = {
        'input_ids': sentence_ids + candidate_ids,
        'attention_mask': [1] * len(sentence_ids + candidate_ids),
        'target_index': len(tokenized_parts[0]) + len(tokenized_parts[1]),
        'label_index': candidate_last_token_index(label_index),
        'candidate_indices': [candidate_last_token_index(i)  for i in range(len(candidates))]
    }
    return item


# Load and prepare the dataset
dataset = load_lexical_substitution_dataset()
dataset = DatasetDict(
    train=dataset['train'].select(range(8)),
    test=dataset['test'].select(range(4)),
)
dataset = dataset.map(prepare_input, remove_columns=dataset['train'].column_names)
