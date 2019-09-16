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


class BertSpanPredReader(DatasetReader):
    """
    BertData Reader
    """

    def __init__(self,
                 bert_tokenizer: BertTokenizer,
                 lazy: bool = False,
                 example_filter=None,
                 max_query_length=64) -> None:

        # max_l indicate the max length of each individual sentence.
        super().__init__(lazy=lazy)
        self._example_filter = example_filter   # If filter equals True, then we delete this example
        self.bert_tokenizer: BertTokenizer = bert_tokenizer
        self.max_query_length = max_query_length

    @overrides
    def _read(self, data_list):
        logger.info("Reading query sentence instances from upstream sampler")
        for example in data_list:

            if self._example_filter is not None and self._example_filter(example):
                continue

            query_tokens = self.bert_tokenizer.tokenize(example['o_query'])
            query_tokens = query_tokens[:self.max_query_length]
            example['query_c_tokens'] = query_tokens

            yield self.text_to_instance(example)

    @overrides
    def text_to_instance(self,  # type: ignore
                         example) -> Instance:

        fields: Dict[str, Field] = {}

        joint_tokens_seq = ['[CLS]'] + example['query_c_tokens'] + ['[SEP]'] + example['context_c_tokens'] + ['[SEP]']
        assert len(joint_tokens_seq) < 512

        text1_len = len(example['query_c_tokens']) + 2
        text2_len = len(example['context_c_tokens']) + 1

        segments_ids = [0 for _ in range(text1_len)] + [1 for _ in range(text2_len)]

        joint_tokens_ids = self.bert_tokenizer.convert_tokens_to_ids(joint_tokens_seq)
        assert len(joint_tokens_ids) == len(segments_ids)

        fields['paired_sequence'] = BertIndexField(np.asarray(joint_tokens_ids, dtype=np.int64))
        fields['paired_segments_ids'] = BertIndexField(np.asarray(segments_ids, dtype=np.int64))

        # This text span is begin inclusive and end exclusive.
        text1_span = (1, 1 + len(example['query_c_tokens'])) # End is exclusive (important for later use)
        text2_span = (text1_span[1] + 1, text1_span[1] + 1 + len(example['context_c_tokens']))

        fields['bert_s1_span'] = SpanField(text1_span[0], text1_span[1], fields['paired_sequence'])
        fields['bert_s2_span'] = SpanField(text2_span[0], text2_span[1], fields['paired_sequence'])
        # fields['bert_s2_span'] = SpanField(text2_span)
        # fields['bert_s1_span'] = MetadataField(text1_span)
        # fields['bert_s2_span'] = MetadataField(text2_span)

        # However, the ground truth span is begin and end both inclusive
        fields['gt_span'] = SpanField(example['start_position'], example['end_position'], fields['paired_sequence'])

        fields['fid'] = IdField(example['fid'])
        fields['uid'] = IdField(example['uid'])

        return Instance(fields)