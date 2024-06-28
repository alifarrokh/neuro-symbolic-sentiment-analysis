import os
import re
import random
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from bs4 import BeautifulSoup
from utils import read_lines


# Constants
COINCO_XML_PATH = 'datasets/CoInCo/coinco.xml'
STS_CSV_PATH = 'datasets/STS-Gold.csv'
LST_DIR = 'datasets/LST/'


def load_rotten_tomatoes():
    dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes')
    return dataset, 2


def load_sts(test_ratio=0.2, seed=48):
    """Load STS dataset"""
    df = pd.read_csv(STS_CSV_PATH)
    df = df.drop('id', axis=1)
    df['polarity'] = df['polarity'].apply(lambda p: int(int(p) == 4))
    df['tweet'] = df['tweet'].apply(str.strip)
    df = df.rename(columns={'polarity': 'label', 'tweet': 'text'})

    # Split train and test
    data = df.to_dict('records')
    random.seed(seed)
    random.shuffle(data)
    n_test = int(len(data) * test_ratio)
    data_train = data[:-n_test]
    data_test = data[-n_test:]

    dataset = DatasetDict(
        train=Dataset.from_list(data_train),
        test=Dataset.from_list(data_test),
    )
    return dataset, 2


def parse_lst_candidate(line):
    """Parse each line of lst.gold.candidates"""
    target_word, candidates = line.split('::')
    target_word = target_word.strip()
    candidates = candidates.strip().split(';')
    candidates = [re.sub('\s+', ' ', c) for c in candidates]
    return target_word, candidates


def parse_lst_label(line):
    """Parse each line of lst_all.gold"""
    identifier, candidates = line.split('::')
    _target_word, id = identifier.strip().split(' ')
    first_candidate = candidates.split(';')[0].strip()
    label = first_candidate[:first_candidate.rfind(' ')]
    return int(id), label.strip()


def preprocess_lst_sentence(s):
    """Clean a given sentence of LST dataset"""
    chars_to_replace = [
        ('$ ', '$'),
        (' %', '%'),
        ('&amp;', '&'),
        ('&gt;', '>'),
        ('`', '\''),
        ('Â¡Â°', '"'),
        ('Â¡Â±', '"'),
        ('Â¡Â¯', '\''),
        ('Â', '\''),
        (u'\u2013', '-'),
        (u'\u2014', '-'),
        (u'\u2018', '"'),
        (u'\u2019', '"'),
        (u'\u201c', '"'),
        (u'\u201d', '"'),
        (u'\u2022', ''),
    ]
    for c_old, c_new in chars_to_replace:
        s = s.replace(c_old, c_new)

    chars_to_remove = u'^~\u0080\u0091\u0092\u0093\u0094\u0096\u0099\u00bb'
    for c in chars_to_remove:
        s = s.replace(c, '')

    # Fix apostrophes
    chars_to_replace = [
        (" 's", "'s"),
        (" n't", "n't"),
        (" 'd", "'d"),
        ("i'd", "I'd"),
        (" 've", "'ve"),
        (" 'll", "'ll"),
        (" 're", "'re"),
        (" 'm ", "'m "),
        (" 't", "'t"),
        (" ' s ", "'s "),
    ]
    for c_old, c_new in chars_to_replace:
        s = s.replace(c_old, c_new)

    s = re.sub('[\s\r\n]+', ' ', s).strip()
    return s


def sample_lst_candidates(candidates, label, max_candidates):
    """Randomly sample candidates from LST dataset"""
    if label in candidates:
        candidates.remove(label)

    # Sample candidates
    random.shuffle(candidates)
    candidates = candidates[:max_candidates-1]

    # Append the true label and shuffle again
    candidates.append(label)
    random.shuffle(candidates)
    return candidates


def load_lst(max_candidates=15, seed=48):
    """Load LST dataset"""
    candidates_path = os.path.join(LST_DIR, 'lst.gold.candidates')
    labels_path = os.path.join(LST_DIR, 'lst_all.gold')
    sentences_path = os.path.join(LST_DIR, 'lst_all_edited.xml')
    random.seed(seed)

    # Load candidates
    lines = read_lines(candidates_path)
    lines = [parse_lst_candidate(l) for l in lines]
    candidates_dict = {t:candidates for t, candidates in lines}

    # Load labels
    lines = read_lines(labels_path)
    lines = [parse_lst_label(l) for l in lines]
    labels = {id:label for id, label in lines}

    # Load example sentences
    with open(sentences_path, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    xml_data = BeautifulSoup(xml_data, "xml")

    all_sentences = []
    for lexelt in xml_data.find_all('lexelt'):
        # Note that this target_word is a reduced form (lemma) of the original target word in the sentence (wordform)
        target_word = lexelt['item']
        target_word = target_word if target_word.count('.') < 2 else target_word[:-2]

        # Parse sentences
        sentences = lexelt.find_all('instance')
        sentences = [{'id': int(s['id']), 'sentence': preprocess_lst_sentence(s.find('context').decode_contents())} for s in sentences]

        # Extract the original target_token (its wordform)
        sentences = [s | {'target_token': re.split('<head>|</head>', s['sentence'])[1]} for s in sentences]

        sentences = [s for s in sentences if s['id'] in labels] # Ignore sentences that have no label
        sentences = [s | {
            'id': f"lst_{s['id']}",
            # 'target_word': target_word,
            # 'pos': target_word.split('.')[1],
            # 'target_token': target_word.split('.')[0],
            'label': labels[s['id']],
            'candidates': sample_lst_candidates(candidates_dict[target_word].copy(), labels[s['id']], max_candidates)
        } for s in sentences]
        all_sentences.extend(sentences)

    dataset = Dataset.from_list(all_sentences)
    return dataset


def preprocess_coinco_sentence(s):
    """
    Preprocessing CoInCo sentences
    Note: The order of operations is imortant.
    """
    strs_to_replace = [
        ('”', '"'),
        ('“', '"'),
        ('’', '\''), # it is not replaced with " because it is often used as apostrophe
        ('‘', '"'),
        ('`', '"'),
        ('...', ' '),
        ('\'\'', '"'),
        ('``', '"'),
        ('–', ' - '),
        ('ç', 'c'),
    ]
    for s_old, s_new in strs_to_replace:
        s = s.replace(s_old, s_new)

    # Chars to remove
    s = re.sub('[>_]', '', s)

    # Remove , between digits
    s = re.sub(r'(\d),(\d)', r'\g<1>\g<2>', s)

    # Add space between and after some punctuation marks
    add_space_chars = '"?!,;…/()'
    for c in add_space_chars:
        s = s.replace(c, f' {c} ')

    # dolon (:) is a bit special as it can in forms like 12:30
    s = re.sub(r'(\D):(\D)', r'\g<1> : \g<2>', s)

    # dot (.) is a bit more special, add a space if ...
    # TODO: The first rule should not affect abbreviation like No., a.m., Feb., Mr., Corp., Inc., etc.
    s = re.sub(r'([a-z\'%\(\)])\.', r'\g<1> . ', s) # if . follows a lowercase char or another punctuation mark
    s = re.sub(r'(\d)\.(\D)', r'\g<1> . \g<2>', s) # if . is at the end of a number

    # Fix repeated spaces
    s = re.sub('[\s\r\n]+', ' ', s).strip()

    # Fix possessive apostrophes
    s = s.replace(" 's ", "'s ")

    # Replace all quotation marks with "
    s = s.replace(" '", ' "')
    s = re.sub(r"([^sn])' ", r'\g<1> "', s)
    return s


def load_coinco(max_candidates=15, seed=48):
    """Load CoInco dataset"""
    random.seed(seed)

    # Load the XML file
    with open(COINCO_XML_PATH, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    xml_data = BeautifulSoup(xml_data, "xml")

    # Parse sentencces
    sents = xml_data.find_all('sent')
    
    all_sentences = []
    for sent in sents:
        sentence = preprocess_coinco_sentence(sent.find('targetsentence').text)

        # Used for adding <head> tags
        temp_left = ''
        temp_right = sentence

        tokens = sent.find_all('token')
        tokens = [t for t in tokens if t['id'] != "XXX"]
        for token in tokens:
            id = token['id']
            target_token = token['wordform']

            # Add <head> tag
            left, right = temp_right.split(
                target_token,
                maxsplit=1, # Avoid over-splitting when there are more than one target_token in the sentence
            )
            temp_left += left
            temp_right = right
            tagged_sentence = f'{temp_left}<head>{target_token}</head>{temp_right}'
            temp_left += target_token
            assert re.sub('<head>|</head>', '', tagged_sentence) == sentence, "A problem occurred while adding the <head> tag."

            # Parse candidates
            candidates = token.find_all('subst')
            candidates = [(c['lemma'], int(c['freq'])) for c in candidates]
            candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
            label = candidates.pop(0)[0]

            # Sample candidates
            candidates = [c[0] for c in candidates]
            random.shuffle(candidates)
            candidates = candidates[:max_candidates-1]

            # Append the true label and shuffle again
            candidates.append(label)
            random.shuffle(candidates)

            all_sentences.append({
                'id': f'coinco_{id}',
                'sentence': tagged_sentence,
                'target_token': target_token,
                'label': label,
                'candidates': candidates
            })

    dataset = Dataset.from_list(all_sentences)
    return dataset


def load_lexical_substitution_dataset(test_ratio=0.3, max_candidates=15, seed=48):
    """Merges LST & CoInCo datasets"""
    lst_dataset = load_lst(max_candidates=max_candidates, seed=seed)
    coinco_dataset = load_coinco(max_candidates=max_candidates, seed=seed)
    dataset = concatenate_datasets([lst_dataset, coinco_dataset])
    return dataset.train_test_split(test_size=test_ratio, seed=seed)


if __name__ == '__main__':
    print(load_lexical_substitution_dataset())
