# This is created for building data from bert tokenizer.
from typing import Dict, List, Tuple
import json

from overrides import overrides
import logging
import numpy as np
import random

from allennlp.data.fields import MetadataField, SpanField

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.fields import Field, LabelField
from allennlp.data.instance import Instance

from data_utils.customized_field import IdField, BertIndexField

# from pathlib import Path
from pathlib import Path
import config

from pytorch_pretrained_bert import BertTokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BertSeparateSeqReader(DatasetReader):
    """
    BertData Reader
    """

    def __init__(self,
                 bert_tokenizer: BertTokenizer,
                 lazy: bool = False,
                 example_filter=None,
                 s1_l=256,
                 s2_l=64) -> None:

        # max_l indicate the max length of each individual sentence.
        super().__init__(lazy=lazy)
        self._example_filter = example_filter   # If filter equals True, then we delete this example
        self.s1_l = s1_l  # This is the max_length of whole sequence
        self.s2_l = s2_l  # This is the max_length of whole sequence
        self.bert_tokenizer: BertTokenizer = bert_tokenizer

    @overrides
    def _read(self, data_list):
        logger.info("Reading query-context instances from upstream sampler")
        for example in data_list:

            label = example["label"]

            if label == '-':
                continue

            if self._example_filter is not None and self._example_filter(example):
                continue

            # We use binary parse here
            # first element is the sentence and the second is the upstream semantic relatedness score.

            evid = example['evid']
            sentence1 = ' '.join(e[0] for e in evid)
            sentence2: str = example["claim"]
            # sentence2: str = example["evid"]

            if len(sentence1) == 0:
                sentence1 = "EEMMPPTTYY"

            pid = str(example['id'])

            yield self.text_to_instance(sentence1, sentence2, pid, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         s1: str,  # Important type information
                         s2: str,
                         pid: str,
                         selection_label: str = None) -> Instance:

        fields: Dict[str, Field] = {}

        tokenized_text1 = self.bert_tokenizer.tokenize(s1)
        tokenized_text2 = self.bert_tokenizer.tokenize(s2)

        # _truncate_seq_pair(tokenized_text1, tokenized_text2, self.max_l)
        tokenized_text1 = tokenized_text1[:self.s1_l]
        tokenized_text2 = tokenized_text2[:self.s2_l]

        s1_tokens_seq = ['[CLS]'] + tokenized_text1
        s2_tokens_seq = ['[CLS]'] + tokenized_text2

        # text1_len = len(tokenized_text1) + 1
        # text2_len = len(tokenized_text2) + 1

        # segments_ids = [0 for _ in range(text1_len)] + [1 for _ in range(text2_len)]

        s1_tokens_ids = self.bert_tokenizer.convert_tokens_to_ids(s1_tokens_seq)
        s2_tokens_ids = self.bert_tokenizer.convert_tokens_to_ids(s2_tokens_seq)

        fields['s1_sequence'] = BertIndexField(np.asarray(s1_tokens_ids, dtype=np.int64))
        fields['s2_sequence'] = BertIndexField(np.asarray(s2_tokens_ids, dtype=np.int64))

        text1_span = (1, len(tokenized_text1)) # End is exclusive (important for later use)
        text2_span = (1, len(tokenized_text2))

        fields['bert_s1_span'] = SpanField(text1_span[0], text1_span[1], fields['s1_sequence'])
        fields['bert_s2_span'] = SpanField(text2_span[0], text2_span[1], fields['s2_sequence'])

        if selection_label:
            fields['label'] = LabelField(selection_label, label_namespace='labels')

        # assert fid is not None
        assert pid is not None
        # fields['fid'] = IdField(fid)
        fields['pid'] = IdField(pid)

        return Instance(fields)
