# This is created for building data from bert tokenizer.
from typing import Dict, List, Tuple
import json

from overrides import overrides
import logging
import numpy as np
import random

from allennlp.data.fields import MetadataField

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.fields import Field, LabelField
from allennlp.data.instance import Instance

from data_utils.customized_field import IdField, BertIndexField

# from pathlib import Path
from pathlib import Path
import config

from pytorch_pretrained_bert import BertTokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BertReaderSentM(DatasetReader):
    """
    BertData Reader
    """

    def __init__(self,
                 bert_tokenizer: BertTokenizer,
                 lazy: bool = False,
                 example_filter=None,
                 query_l=64,
                 context_l=256,
                 max_l=300) -> None:

        # max_l indicate the max length of each individual sentence.
        super().__init__(lazy=lazy)
        self._example_filter = example_filter   # If filter equals True, then we delete this example
        self.query_l = query_l  # This is the max_length of whole sequence
        self.context_l = context_l  # This is the max_length of whole sequence
        self.max_l = max_l
        self.bert_tokenizer: BertTokenizer = bert_tokenizer

    @overrides
    def _read(self, data_list):
        logger.info("Reading query sentence instances from upstream sampler")
        for example in data_list:

            label = example["label"]

            if label == '-':
                continue

            if self._example_filter is not None and self._example_filter(example):
                continue

            # We use binary parse here
            # first element is the sentence and the second is the upstream semantic relatedness score.

            sentence1: str = example["question"]    # Question go first according to BERT paper.
            # truncate premise
            sentence2: str = example["sentence"]

            if len(sentence1) == 0:
                sentence1 = "EEMMPPTTYY"

            pid = str(example['selection_id'])

            yield self.text_to_instance(sentence1, sentence2, pid, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sent1: str,  # Important type information
                         sent2: str,
                         pid: str = None,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {}

        tokenized_text1 = self.bert_tokenizer.tokenize(sent1)
        tokenized_text2 = self.bert_tokenizer.tokenize(sent2)

        # _truncate_seq_pair(tokenized_text1, tokenized_text2, self.max_l)
        tokenized_text1 = tokenized_text1[:self.query_l]
        tokenized_text2 = tokenized_text2[:(self.max_l - len(tokenized_text1))]

        joint_tokens_seq = ['[CLS]'] + tokenized_text1 + ['[SEP]'] + tokenized_text2 + ['[SEP]']
        text1_len = len(tokenized_text1) + 2
        text2_len = len(tokenized_text2) + 1
        segments_ids = [0 for _ in range(text1_len)] + [1 for _ in range(text2_len)]

        joint_tokens_ids = self.bert_tokenizer.convert_tokens_to_ids(joint_tokens_seq)
        assert len(joint_tokens_ids) == len(segments_ids)

        fields['paired_sequence'] = BertIndexField(np.asarray(joint_tokens_ids, dtype=np.int64))
        fields['paired_segments_ids'] = BertIndexField(np.asarray(segments_ids, dtype=np.int64))

        text1_span = (1, 1 + len(tokenized_text1)) # End is exclusive (important for later use)
        text2_span = (text1_span[1] + 1, text1_span[1] + 1 + len(tokenized_text2))

        fields['bert_s1_span'] = MetadataField(text1_span)
        fields['bert_s2_span'] = MetadataField(text2_span)

        if label:
            fields['label'] = LabelField(label, label_namespace='labels')

        if pid:
            fields['pid'] = IdField(pid)

        return Instance(fields)
