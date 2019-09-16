# This is created for building data from bert tokenizer.
from typing import Dict, List, Tuple
import json

from overrides import overrides
import logging
import numpy as np

from allennlp.data.fields import MetadataField, SpanField

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.fields import Field, LabelField
from allennlp.data.instance import Instance

from data_utils.customized_field import IdField, BertIndexField


from pytorch_pretrained_bert import BertTokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BertContentSelectionReader(DatasetReader):
    """
    BertData Reader
    """

    def __init__(self,
                 bert_tokenizer: BertTokenizer,
                 lazy: bool = False,
                 example_filter=None,
                 query_l=64,
                 context_l=300,
                 max_l=128,
                 is_paired=True) -> None:

        # max_l indicate the max length of each individual sentence.
        super().__init__(lazy=lazy)
        self._example_filter = example_filter   # If filter equals True, then we delete this example
        self.query_l = query_l  # This is the max_length of whole sequence
        self.context_l = context_l  # This is the max_length of whole sequence
        self.bert_tokenizer: BertTokenizer = bert_tokenizer
        self.is_paired = is_paired
        self.max_l = max_l

    @overrides
    def _read(self, data_list):
        logger.info("Reading query-context instances from upstream sampler")
        for example in data_list:

            selection_label = example["s_labels"]

            if selection_label == '-':
                continue

            if self._example_filter is not None and self._example_filter(example):
                continue

            # We use binary parse here
            # first element is the sentence and the second is the upstream semantic relatedness score.

            query: str = example["query"]    # Question go first according to BERT paper.
            # truncate premise
            context: str = example["context"]

            # print(example)
            retain_item = example["sid"]  # Change this to some other name for flexibility.

            assert len(query) != 0
            assert len(context) != 0

            fid = str(example['fid'])
            oid = str(example['cid'])

            yield self.text_to_instance(query, context, fid, oid, retain_item, selection_label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         query: str,  # Important type information
                         context: str,
                         fid: str = None,
                         oid: str = None,
                         retain_item: str = None,
                         selection_label: str = None) -> Instance:

        if self.is_paired:
            fields: Dict[str, Field] = {}

            tokenized_text1 = self.bert_tokenizer.tokenize(query)
            tokenized_text2 = self.bert_tokenizer.tokenize(context)

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

            text1_span = (1, 1 + len(tokenized_text1))  # End is exclusive (important for later use)
            text2_span = (text1_span[1] + 1, text1_span[1] + 1 + len(tokenized_text2))

            fields['bert_s1_span'] = SpanField(text1_span[0], text1_span[1], fields['paired_sequence'])
            fields['bert_s2_span'] = SpanField(text2_span[0], text2_span[1], fields['paired_sequence'])

            if selection_label:
                fields['label'] = LabelField(selection_label, label_namespace='labels')

            assert fid is not None
            assert oid is not None
            fields['fid'] = IdField(fid)
            fields['oid'] = IdField(oid)
            fields['item'] = IdField(retain_item)

            return Instance(fields)

        else:
            fields: Dict[str, Field] = {}

            tokenized_text1 = self.bert_tokenizer.tokenize(query)
            tokenized_text2 = self.bert_tokenizer.tokenize(context)

            # _truncate_seq_pair(tokenized_text1, tokenized_text2, self.max_l)
            tokenized_text1 = tokenized_text1[:self.query_l]
            tokenized_text2 = tokenized_text2[:self.context_l]

            s1_tokens_seq = ['[CLS]'] + tokenized_text1
            s2_tokens_seq = ['[CLS]'] + tokenized_text2

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

            assert fid is not None
            assert oid is not None
            fields['fid'] = IdField(fid)
            fields['oid'] = IdField(oid)
            fields['item'] = IdField(retain_item)

            return Instance(fields)
