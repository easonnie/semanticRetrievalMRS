import unicodedata

import spacy

nlp = spacy.load('en')

nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')


def spacy_tokenize(text, uncase=False):
    text = normalize(text)
    t_text = nlp(text)
    if not uncase:
        return [c_token.text for c_token in t_text]
    else:
        return [c_token.text.lower() for c_token in t_text]


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)
