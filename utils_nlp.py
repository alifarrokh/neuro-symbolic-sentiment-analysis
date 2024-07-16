import os
import random
import nltk
import wn


# Global singletones
wordnet = None


def get_word_synonyms(word, pos, limit: int = None):
    assert pos in 'anvr', "Invalid POS tag"
    global wordnet

    # Load the wordnet
    if wordnet is None:
        wordnet = wn.Wordnet('oewn:2023')

    # Find the list of synonyms
    synonyms = {word}
    synsets = wordnet.synsets(word, pos=pos)
    for syn in synsets:
        for sense in syn.senses():
            for lemma in sense.synset().lemmas():
                synonyms.add(lemma)
    synonyms.remove(word)
    synonyms = list(synonyms)

    # Return a subset
    if limit is not None:
        random.shuffle(synonyms)
        synonyms = synonyms[:limit]

    return synonyms


def upenn_to_wn_tag(upenn_tag: str):
    """
    Converts a UPenn tag to one of the four accepted tags by WordNet (a, v, n, r).
    Returns None if the tag is irrelevant in the context of WordNet.
    Raises an exception if the input tag is unavailable in UPenn tagset.
    More info on UPenn tags:
    ```
    import nltk
    nltk.help.upenn_tagset()
    ```
    """
    punctuation = "$`'(),-.:"
    irrelevant_tags = list(punctuation) + [
        'CC', # coordinating conjunction (and, both, ...)
        'CD', # numeral, cardinal
        'DT', # determiner
        'EX', # existential there
        'FW', # foreign word
        'IN', # preposition or conjunction, subordinating
        'LS', # list item marker
        'MD', # modal auxiliary
        'PDT', # pre-determiner
        'POS', # genitive marker
        'PRP', # pronoun, personal (hers, herself, him, himself, ...)
        'PRP$', # pronoun, possessive (her, his, mine, ...)
        'RP', # particle (aboard, about, across, upon, ...)
        'SYM', # symbol
        'TO', # "to" as preposition or infinitive marker
        'UH', # interjection (Goodbye, Goody, Gosh, Wow, ...)
        'WDT', # WH-determiner (that, what, whatever, ...)
        'WP', # WH-pronoun (that, what, whatever, whatsoever, ...)
        'WP$', # WH-pronoun, possessive (whose)
        'WRB', # Wh-adverb (how, however, whence, whenever, ...)
    ]
    adj_tags = [
        'JJ', # adjective or numeral, ordinal
        'JJR', # adjective, comparative
        'JJS', # adjective, superlative
    ]
    noun_tags = [
        'NN', # noun, common, singular or mass
        'NNP', # noun, proper, singular
        'NNPS', # noun, proper, plural
        'NNS', # noun, common, plural
    ]
    adv_tags = [
        'RB', # adverb
        'RBR', # adverb, comparative
        'RBS', # adverb, superlative
    ]
    verb_tags = [
        'VB', # verb, base form
        'VBD', # verb, past tense
        'VBG', # verb, present participle or gerund
        'VBN', # verb, past participle
        'VBP', # verb, present tense, not 3rd person singular
        'VBZ', # verb, present tense, 3rd person singular
    ]
    if upenn_tag in adj_tags:
        return 'a'
    elif upenn_tag in noun_tags:
        return 'n'
    elif upenn_tag in adv_tags:
        return 'r'
    elif upenn_tag in verb_tags:
        return 'v'
    elif upenn_tag in irrelevant_tags:
        return None
    else:
        raise ValueError('The given tag is not a valid UPenn tag.')


if __name__ == '__main__':
    # Download WordNet database
    os.system('python -m wn download oewn:2023')

    # Download NLTK packages
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')
    nltk.download('punkt')
