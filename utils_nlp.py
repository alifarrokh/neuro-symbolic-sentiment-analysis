import random
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
