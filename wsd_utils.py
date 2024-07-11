from dataclasses import dataclass
from typing import Optional
import re
import string
import random
import numpy as np
import nltk
import torch
from utils_nlp import upenn_to_wn_tag, get_word_synonyms


def clean_selected_word(w):
    """Remove punctuation and strip"""
    w = w.translate(str.maketrans('', '', string.punctuation))
    w = re.sub('\s', ' ', w).strip()
    return w


@dataclass
class WordInfo:
    """Keeps the information of a word in a sentence"""
    sentence: str
    word: str
    index: int
    att_weight: float
    embedding: np.ndarray
    pos: Optional[str] = None
    sense_diversity: Optional[float] = None

    def get_synonyms(self, limit: int = None):
        if not hasattr(self, '_synonyms') or self._synonyms is None:
            self._synonyms = get_word_synonyms(self.word, self.pos)
        if limit and len(self._synonyms) > limit:
            random.shuffle(self._synonyms)
            return self._synonyms[:limit]
        return self._synonyms


def get_word_infos(sentence, tokenizer, data_collator, sa_model, with_pos=False) -> list[WordInfo]:
    """Returns a list of WordInfo objects for a given sentence"""
    words = sentence.split(" ")

    # Get POS tags
    if with_pos:
        words_tagged = nltk.pos_tag(words)
        pos_tags = [upenn_to_wn_tag(t) for _w,t in words_tagged]

    # Tokenize
    tokenized = tokenizer.encode_plus(sentence)
    model_input = data_collator([tokenized])

    # Extract attention weights from model
    with torch.no_grad():
        model_input = {k:v.to(sa_model.device) for k,v in model_input.items()}
        _logits, embeddings, attention_weights = sa_model(**model_input, return_analysis_info=True)
    embeddings = embeddings.squeeze().detach().cpu().numpy()[1:-1, :]
    embeddings = [vec for vec in embeddings] # Convert to list of token embeddings
    attention_weights = attention_weights.squeeze().detach().tolist()
    attention_weights = attention_weights[1:-1] # Exclude <s> and </s>

    # Match words with their attention weights and embeddings
    word_ids = tokenized.word_ids()[1:-1]
    word_infos = []
    for i in range(len(words)):
        weights = []
        word_embeddings = []
        while len(word_ids) > 0 and word_ids[0] == i:
            weights.append(attention_weights.pop(0))
            word_embeddings.append(embeddings.pop(0))
            word_ids.pop(0)

        word_info = WordInfo(
            sentence=sentence,
            word=clean_selected_word(words[i]),
            index=i,
            att_weight=np.max(weights),
            embedding=np.array(word_embeddings).mean(axis=0),
            pos=pos_tags[i] if with_pos else None,
        )
        word_infos.append(word_info)
    return word_infos


def compute_sense_diversity(
    word: WordInfo,
    tokenizer,
    data_collator,
    sa_model,
):
    """Computes the sense diversity of a word"""
    words = word.sentence.split(' ')
    synonyms = word.get_synonyms()
    if len(synonyms) == 0:
        return 0

    # Create new sentences by replacing the word with its synonyms
    sentences = []
    syn_word_len = []
    for syn in synonyms:
        new_words = words.copy()
        new_words[word.index] = syn
        syn_word_len.append(len(syn.split(' ')))
        sentences.append(' '.join(new_words))

    # Extract word embeddings for each synonym
    sentence_word_infos = [get_word_infos(s, tokenizer, data_collator, sa_model) for s in sentences]
    syn_word_infos = [sentence_word_infos[i][word.index: word.index+syn_word_len[i]] for i in range(len(synonyms))] # Keep word infos related to each synonym
    syn_embedding = [np.array([w.embedding for w in w_infos]).mean(axis=0) for w_infos in syn_word_infos]

    # Compute sense diversity
    euclidean_distances = [np.linalg.norm(word.embedding - e) for e in syn_embedding]
    sense_diversity = np.mean(euclidean_distances)
    return sense_diversity
