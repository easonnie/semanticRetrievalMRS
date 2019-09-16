We need to convert the original file in a way that is easy to process later.

We save the document into sqiltdict and we append a field to indicator the processed element.

sentences: List of sentences (list of tokens).
paragraph_tags: List of numbers: 0, 0, 1, 1 indicate the paragraph number of sentence
hyperlink: List of hyperlinks (list of hyperlink: start_token, end_token, title w.r.t the current sentence)
pos: List of pos tags (list of pos tag)     From spacy