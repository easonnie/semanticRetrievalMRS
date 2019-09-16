"""Evaluate the accuracy of the retriever module."""
import unicodedata

import regex as re
import spacy
from typing import List
from sacremoses import MosesDetokenizer
from utils.text_process_tool import spacy_tokenize, normalize

md = MosesDetokenizer(lang='en')


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answers, retrieved_text, match='string', tokenized: bool = False):
    """Check if retrieved_text contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    if not isinstance(answers, list):
        answers = [answers]

    if match == 'string':
        if tokenized:
            text = md.detokenize(retrieved_text)
            t_text = retrieved_text
        else:
            text = retrieved_text
            t_text = spacy_tokenize(retrieved_text, uncase=True)

        for single_answer in answers:
            single_answer = spacy_tokenize(single_answer, uncase=True)

            for i in range(0, len(t_text) - len(single_answer) + 1):
                if single_answer == t_text[i: i + len(single_answer)]:
                    return True

        for single_answer in answers:   # If raw covered.
            if single_answer in text:
                return True

    elif match == 'regex':
        if tokenized:
            text = md.detokenize(retrieved_text)
        else:
            text = retrieved_text

        # Answer is a regex
        single_answer = normalize(answers[0])
        if regex_match(text, single_answer):
            return True
    return False


def utest_normal():
    paragraph = "I'm name is eason"
    answer = "Eason"
    print(has_answer(answer, paragraph, match='string', tokenized=False))


def utest_regex():
    # {"question": "How deep is Crater Lake?",
    #  "answer": ["1\\s?,\\s?932 feet|1,?949\\s*f(ee|oo)?t|594\\s*m|593 m|593\\.0|594\\.0552"]}
    # paragraph = "When is Fashion week in NYC?"
    # paragraph = "1 , 932 feet"
    # paragraph = "120  km/h"
    paragraph = ['3', ',', '390', 'km']

    # answer = "Sept?(ember)?|Feb(ruary)?"
    # answer = "1\\s?,\\s?932 feet|1,?949\\s*f(ee|oo)?t|594\\s*m|593 m|593\\.0|594\\.0552"
    # answer = "120\\s*km/?h|75\\s*mph"

    answer = "diameter.*(4,?21[0-9]\\s*miles|6[78][0-9][0-9]\\s*(km|kilometers))|radius.*(2,?106\\s*miles|3,?390\\s*(km|kilometers))|3390km|3390 km|3\\,389\\.5 km"

    print(has_answer(answer, paragraph, match='regex', tokenized=True))


if __name__ == '__main__':
    utest_regex()