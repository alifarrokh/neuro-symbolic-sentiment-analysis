from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding
)
from sentiment_analysis_model import RobertaForSentimentAnalysis
from lexical_substitution_model import (
    LexicalSubstitutionInputFormatter,
    LexicalSubstitutionDataCollator,
    RobertaForLexicalSubstitution
)
from load_datasets import load_rotten_tomatoes
from wsd_utils import get_word_infos, compute_sense_diversity


# Config
compute_original_test_accuracy = False # Current = 88.274

# Hyper-parameters
I = 5 # Max number of selected words with highest attention weights
J = 2 # Max number of selected words with highest sense diversity
K = 15 # Max number of candidates for lexical substitution

# Recognize the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load LS model
ls_tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
ls_input_formatter = LexicalSubstitutionInputFormatter(ls_tokenizer)
ls_data_collator = LexicalSubstitutionDataCollator(ls_tokenizer)
ls_model = RobertaForLexicalSubstitution.from_pretrained('exps/ls/exp1/checkpoint-2016', device_map=device)
ls_model.eval()

# Load SA model
sa_tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
sa_data_collator = DataCollatorWithPadding(tokenizer=sa_tokenizer)
sa_model = RobertaForSentimentAnalysis.from_pretrained('exps/roberta-rt-han+a/checkpoint-1166', device_map=device)
sa_model.eval()

# Load SA dataset
sa_dataset, num_labels = load_rotten_tomatoes()
sa_dataset = sa_dataset['test']

# Compute the original test accuracy
if compute_original_test_accuracy:
    sa_dataset_tokenized = sa_dataset.map(lambda item: sa_tokenizer(item['text'], truncation=True), remove_columns=['text'])
    sa_dataloader = DataLoader(sa_dataset_tokenized, batch_size=4, collate_fn=sa_data_collator)

    labels = []
    preds = []
    with torch.no_grad():
        for batch in tqdm(sa_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}
            labels.extend(batch['labels'].tolist())
            preds.extend(sa_model(**batch).logits.argmax(axis=-1).tolist())
    accuracy = (np.array(labels) == np.array(preds)).sum() / len(preds)
    print(f'Original Accuracy: {accuracy*100:.3f}')

# Create the improved dataset by substituting ambiguous words 
new_dataset_items = []
for item in tqdm(sa_dataset):
    # Get word infos
    word_infos = get_word_infos(item['text'], sa_tokenizer, sa_data_collator, sa_model, with_pos=True)

    # Select top I words
    word_infos = [w for w in word_infos if w.pos is not None] # Filter irrelevant tags
    word_infos = sorted(word_infos, key=lambda w: w.att_weight, reverse=True) # Sort by weight (descending)
    word_infos = word_infos[:I] # Keep the first I words

    # Compute sense diversities and select the top J words for disambiguation
    sense_diversities = [compute_sense_diversity(word_infos[i], sa_tokenizer, sa_data_collator, sa_model) for i in range(len(word_infos))]
    for i in range(len(sense_diversities)):
        word_infos[i].sense_diversity = sense_diversities[i]
    word_infos = sorted(word_infos, key=lambda w: w.sense_diversity, reverse=True)
    word_infos = [w for w in word_infos if w.sense_diversity > 0] # Filter non-ambiguous words
    selected_words = word_infos[:J]

    # Use the original sentence if there are no substitution candidates
    if len(selected_words) == 0:
        new_dataset_items.append(item)
        continue

    # Find the best substitutions
    substitutions = []
    for word in selected_words:
        candidates = word.get_synonyms(limit=K)

        # Create the sentence
        sentence_words = word.sentence.split(' ')
        words_left, words_right = sentence_words[:word.index], sentence_words[word.index+1:]
        input_sentence = ' '.join(words_left) + f' <head>{word.word}</head> ' + ' '.join(words_right)

        # Format the input
        model_input = {
            'sentence': input_sentence,
            'target_token': word.word,
            'candidates': candidates
        }
        model_input = ls_input_formatter(model_input)
        model_input = ls_data_collator([model_input])

        # Find the best substitution candidate
        with torch.no_grad():
            model_input = {k:v.to(device) for k,v in model_input.items()}
            pred_token_index = ls_model(**model_input).logits[0].item()
        pred_candidate_index = model_input['candidate_indices'][0].tolist().index(pred_token_index)
        selected_substitute = candidates[pred_candidate_index]
        substitutions.append({'word_index': word.index, 'new_word': selected_substitute})

    # Replace the new words
    words = selected_words[0].sentence.split(' ')
    for substitution in substitutions:
        words[substitution['word_index']] = substitution['new_word']
    new_sentence = ' '.join(words)
    new_item = item | {'text': new_sentence}
    new_dataset_items.append(new_item)

# Compute the accuracy on the new dataset
new_sa_dataset = Dataset.from_list(new_dataset_items)
new_sa_dataset_tokenized = new_sa_dataset.map(lambda item: sa_tokenizer(item['text'], truncation=True), remove_columns=['text'])
new_sa_dataloader = DataLoader(new_sa_dataset_tokenized, batch_size=4, collate_fn=sa_data_collator)

labels = []
preds = []
with torch.no_grad():
    for batch in tqdm(new_sa_dataloader):
        labels.extend(batch['labels'].tolist())
        preds.extend(sa_model(**batch).logits.argmax(axis=-1).tolist())
accuracy = (np.array(labels) == np.array(preds)).sum() / len(preds)
print(f'New Accuracy: {accuracy*100:.3f}')
