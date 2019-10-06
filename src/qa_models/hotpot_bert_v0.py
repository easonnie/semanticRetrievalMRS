import os
import random
from enum import Enum
from pathlib import Path

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from pytorch_pretrained_bert.modeling import BertLayerNorm

from data_utils.readers.paired_span_pred_reader import BertPairedSpanPredReader
# from data_utils.readers.span_pred_reader import BertSpanPredReader

import flint.torch_util as torch_util
import torch.nn as nn
from torch.nn.functional import nll_loss
import torch

import config
from evaluation import ext_hotpot_eval
from hotpot_fact_selection_sampler.sampler_s_level_to_qa import get_qa_item_with_upstream_sentence
from neural_modules.model_EMA import EMA, get_ema_gpu_id_list
from span_prediction_task_utils.common_utils import write_to_predicted_fitem, merge_predicted_fitem_to_eitem
from span_prediction_task_utils.squad_utils import preprocessing_squad
from span_prediction_task_utils.span_preprocess_tool import eitems_to_fitems
from utils import common, list_dict_data_tool, save_tool
from allennlp.data.iterators import BasicIterator
from allennlp.nn import util as allen_util
from tqdm import tqdm
import evaluation.squad_eval_v1


def init_bert_weights(module, initializer_range):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.beta.data.normal_(mean=0.0, std=initializer_range)
        module.gamma.data.normal_(mean=0.0, std=initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class BertSpan(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1

    def __init__(self, bert_encoder, num_of_layers=1):
        super(BertSpan, self).__init__()
        self.bert_encoder = bert_encoder
        if num_of_layers == 1:
            self.qa_outputs = nn.Linear(self.bert_encoder.config.hidden_size, 2)  # Should we have dropout here? Later?
            init_bert_weights(self.qa_outputs, initializer_range=0.02)  # Hard code this value

        elif num_of_layers == 2:
            self.output_layer1 = nn.Linear(self.bert_encoder.config.hidden_size, self.bert_encoder.config.hidden_size)
            self.activation = nn.Tanh()
            self.dropout = nn.Dropout(self.bert_encoder.config.hidden_dropout_prob)
            self.output_layer2 = nn.Linear(self.bert_encoder.config.hidden_size, 2)

            self.qa_outputs = nn.Sequential(self.output_layer1, self.activation, self.dropout, self.output_layer2)

            init_bert_weights(self.output_layer1, initializer_range=0.02)
            init_bert_weights(self.output_layer2, initializer_range=0.02)
        else:
            raise ValueError("Number of layers not supported.")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                gt_span=None, mode=ForwardMode.TRAIN):
        sequence_output, _ = self.bert_encoder(input_ids, token_type_ids, attention_mask,
                                               output_all_encoded_layers=False)
        joint_length = allen_util.get_lengths_from_binary_sequence_mask(attention_mask)

        joint_seq_logits = self.qa_outputs(sequence_output)

        # The following line is from AllenNLP bidaf.
        start_logits = allen_util.replace_masked_values(joint_seq_logits[:, :, 0], attention_mask, -1e18)
        # B, T, 2
        end_logits = allen_util.replace_masked_values(joint_seq_logits[:, :, 1], attention_mask, -1e18)

        if mode == BertSpan.ForwardMode.TRAIN:
            assert gt_span is not None
            gt_start = gt_span[:, 0]  # gt_span: [B, 2] -> [B]
            gt_end = gt_span[:, 1]

            start_loss = nll_loss(allen_util.masked_log_softmax(start_logits, attention_mask), gt_start)
            end_loss = nll_loss(allen_util.masked_log_softmax(end_logits, attention_mask), gt_end)
            # We delete squeeze bc it will cause problem when the batch size is 1, and remember the gt_start and gt_end should have shape [B].
            # start_loss = nll_loss(allen_util.masked_log_softmax(start_logits, context_mask), gt_start.squeeze(-1))
            # end_loss = nll_loss(allen_util.masked_log_softmax(end_logits, context_mask), gt_end.squeeze(-1))

            loss = start_loss + end_loss
            return loss
        else:
            return start_logits, end_logits, joint_length


def span_eval(model, data_iter, do_lower_case, fitem_dict, device_num, show_progress, pred_no_answer=True):
    # fitem_dict in the parameter is the original fitem_dict
    output_fitem_dict = {}

    with torch.no_grad():
        model.eval()

        for batch_idx, batch in tqdm(enumerate(data_iter), disable=(not show_progress)):
            batch = allen_util.move_to_device(batch, device_num)
            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            gt_span = batch['gt_span']

            start_logits, end_logits, context_length = model(mode=BertSpan.ForwardMode.EVAL,
                                                             input_ids=paired_sequence,
                                                             token_type_ids=paired_segments_ids,
                                                             attention_mask=att_mask,
                                                             gt_span=gt_span)
            b_fids = batch['fid']
            b_uids = batch['uid']

            write_to_predicted_fitem(start_logits, end_logits, context_length, b_fids, b_uids, gt_span, fitem_dict,
                                     output_fitem_dict, do_lower_case)

    eitem_list, eval_dict = merge_predicted_fitem_to_eitem(output_fitem_dict, None, pred_no_answer=pred_no_answer)
    return eitem_list, eval_dict


def eval_model(model_path, data_file=None, filter_value=0.5):
    seed = 12
    torch.manual_seed(seed)

    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = "bert-base-uncased"
    lazy = False
    forward_size = 16
    batch_size = 32

    do_lower_case = True

    debug = False

    max_pre_context_length = 320
    max_query_length = 64
    doc_stride = 128
    qa_num_of_layer = 2
    s_filter_value = filter_value
    s_top_k = 5

    tag = 'dev'

    print("Potential total length:", max_pre_context_length + max_query_length + 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                              cache_dir=bert_pretrain_path)

    # Load Dataset.
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    test_list = common.load_json(config.TEST_FULLWIKI_FILE)
    train_list = common.load_json(config.TRAIN_FILE)

    if data_file is None:
        dev_sentence_level_results = common.load_jsonl(
            config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/dev_s_level_bert_v1_results.jsonl")
    else:
        dev_sentence_level_results = common.load_jsonl(
            data_file
        )

    test_sentence_level_results = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/test_s_level_bert_v1_results.jsonl")

    train_sentence_level_results = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/train_s_level_bert_v1_results.jsonl")

    dev_fitem_dict, dev_fitem_list, dev_sp_results_dict = get_qa_item_with_upstream_sentence(dev_list,
                                                                                             dev_sentence_level_results,
                                                                                             is_training=False,
                                                                                             tokenizer=tokenizer,
                                                                                             max_context_length=max_pre_context_length,
                                                                                             max_query_length=max_query_length,
                                                                                             filter_value=s_filter_value,
                                                                                             doc_stride=doc_stride,
                                                                                             top_k=s_top_k,
                                                                                             debug_mode=debug)

    test_fitem_dict, test_fitem_list, test_sp_results_dict = get_qa_item_with_upstream_sentence(test_list,
                                                                                                test_sentence_level_results,
                                                                                                is_training=False,
                                                                                                tokenizer=tokenizer,
                                                                                                max_context_length=max_pre_context_length,
                                                                                                max_query_length=max_query_length,
                                                                                                filter_value=s_filter_value,
                                                                                                doc_stride=doc_stride,
                                                                                                top_k=s_top_k,
                                                                                                debug_mode=debug)

    # train_fitem_dict, train_fitem_list, _ = get_qa_item_with_upstream_sentence(train_list, train_sentence_level_results,
    #                                                                            is_training=True,
    #                                                                            tokenizer=tokenizer,
    #                                                                            max_context_length=max_pre_context_length,
    #                                                                            max_query_length=max_query_length,
    #                                                                            filter_value=s_filter_value,
    #                                                                            doc_stride=doc_stride,
    #                                                                            top_k=s_top_k,
    #                                                                            debug_mode=debug)

    if debug:
        dev_list = dev_list[:100]

    span_pred_reader = BertPairedSpanPredReader(bert_tokenizer=tokenizer, lazy=lazy,
                                                example_filter=None)
    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertSpan(bert_encoder, qa_num_of_layer)

    model.load_state_dict(torch.load(model_path))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    iterator = BasicIterator(batch_size=batch_size)

    if tag == 'dev':
        dev_instances = span_pred_reader.read(dev_fitem_list)
        # test_instances = span_pred_reader.read(test_fitem_list)
        eval_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
        # eval_iter = iterator(test_instances, num_epochs=1, shuffle=False)

        cur_eitem_list, cur_eval_dict = span_eval(model, eval_iter, do_lower_case, dev_fitem_dict,
                                                  device_num, show_progress=True, pred_no_answer=True)
        # cur_eitem_list, cur_eval_dict = span_eval(model, eval_iter, do_lower_case, test_fitem_dict,
        #                                           device_num, show_progress=True)

        cur_results_dict = dict()
        cur_results_dict['answer'] = cur_eval_dict
        cur_results_dict['sp'] = dev_sp_results_dict
        # cur_results_dict['sp'] = test_sp_results_dict

        # common.save_json(cur_results_dict, f"{tag}_qa_sp_results_{filter_value}_doctopk_5.json")

        cur_results_dict['p_answer'] = cur_eval_dict
        _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
        # _, metrics = ext_hotpot_eval.eval(cur_results_dict, test_list, verbose=False)

        logging_item = {
            'score': metrics,
        }

        print(data_file)
        print(logging_item)

    elif tag == 'test':
        # dev_instances = span_pred_reader.read(dev_fitem_list)
        test_instances = span_pred_reader.read(test_fitem_list)
        # eval_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
        eval_iter = iterator(test_instances, num_epochs=1, shuffle=False)

        # cur_eitem_list, cur_eval_dict = span_eval(model, eval_iter, do_lower_case, dev_fitem_dict,
        #                                           device_num, show_progress=True)
        cur_eitem_list, cur_eval_dict = span_eval(model, eval_iter, do_lower_case, test_fitem_dict,
                                                  device_num, show_progress=True)

        cur_results_dict = dict()
        cur_results_dict['answer'] = cur_eval_dict
        # cur_results_dict['sp'] = dev_sp_results_dict
        cur_results_dict['sp'] = test_sp_results_dict

        common.save_json(cur_results_dict, f"{tag}_qa_sp_results.json")

        cur_results_dict['p_answer'] = cur_eval_dict
        # _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
        _, metrics = ext_hotpot_eval.eval(cur_results_dict, test_list, verbose=False)

        logging_item = {
            'score': metrics,
        }

        print(logging_item)


def model_go(sent_filter_value, sent_top_k=5):
    seed = 12
    torch.manual_seed(seed)

    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = "bert-base-uncased"
    lazy = False
    forward_size = 32
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_rate = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5
    eval_frequency = 1000

    do_lower_case = True

    debug = False

    max_pre_context_length = 320
    max_query_length = 64
    doc_stride = 128
    qa_num_of_layer = 2
    do_ema = True
    ema_device_num = 1
    # s_filter_value = 0.5
    s_filter_value = sent_filter_value
    # s_top_k = 5
    s_top_k = sent_top_k

    experiment_name = f'hotpot_v0_qa_(s_top_k:{s_top_k},s_fv:{s_filter_value},qa_layer:{qa_num_of_layer})'

    print("Potential total length:", max_pre_context_length + max_query_length + 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                              cache_dir=bert_pretrain_path)

    # Load Dataset.
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    train_list = common.load_json(config.TRAIN_FILE)

    dev_sentence_level_results = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/dev_s_level_bert_v1_results.jsonl")
    train_sentence_level_results = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/train_s_level_bert_v1_results.jsonl")

    dev_fitem_dict, dev_fitem_list, dev_sp_results_dict = get_qa_item_with_upstream_sentence(dev_list,
                                                                                             dev_sentence_level_results,
                                                                                             is_training=False,
                                                                                             tokenizer=tokenizer,
                                                                                             max_context_length=max_pre_context_length,
                                                                                             max_query_length=max_query_length,
                                                                                             filter_value=s_filter_value,
                                                                                             doc_stride=doc_stride,
                                                                                             top_k=s_top_k,
                                                                                             debug_mode=debug)

    train_fitem_dict, train_fitem_list, _ = get_qa_item_with_upstream_sentence(train_list, train_sentence_level_results,
                                                                               is_training=True,
                                                                               tokenizer=tokenizer,
                                                                               max_context_length=max_pre_context_length,
                                                                               max_query_length=max_query_length,
                                                                               filter_value=s_filter_value,
                                                                               doc_stride=doc_stride,
                                                                               top_k=s_top_k,
                                                                               debug_mode=debug)

    # print(len(dev_fitem_list))
    # print(len(dev_fitem_dict))

    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, '_id')

    if debug:
        dev_list = dev_list[:100]
        eval_frequency = 2

    est_datasize = len(train_fitem_list)

    span_pred_reader = BertPairedSpanPredReader(bert_tokenizer=tokenizer, lazy=lazy,
                                                example_filter=None)
    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertSpan(bert_encoder, qa_num_of_layer)

    ema = None
    if do_ema:
        ema = EMA(model, model.named_parameters(), device_num=ema_device_num)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    iterator = BasicIterator(batch_size=batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    print("Total train instances:", len(train_fitem_list))

    num_train_optimization_steps = int(est_datasize / forward_size / gradient_accumulate_step) * \
                                   num_train_epochs

    if debug:
        num_train_optimization_steps = 100

    print("Estimated training size", est_datasize)
    print("Number of optimization steps:", num_train_optimization_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_rate,
                         t_total=num_train_optimization_steps)

    dev_instances = span_pred_reader.read(dev_fitem_list)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    # # # Create Log File
    file_path_prefix = None
    if not debug:
        file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
        # Save the source code.
        script_name = os.path.basename(__file__)
        with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
            out_f.write(it.read())
            out_f.flush()
    # # # Log File end

    for epoch_i in range(num_train_epochs):
        print("Epoch:", epoch_i)

        print("Resampling:")
        train_fitem_dict, train_fitem_list, _ = get_qa_item_with_upstream_sentence(train_list,
                                                                                   train_sentence_level_results,
                                                                                   is_training=True,
                                                                                   tokenizer=tokenizer,
                                                                                   max_context_length=max_pre_context_length,
                                                                                   max_query_length=max_query_length,
                                                                                   filter_value=s_filter_value,
                                                                                   doc_stride=doc_stride,
                                                                                   top_k=s_top_k,
                                                                                   debug_mode=debug)

        random.shuffle(train_fitem_list)
        train_instances = span_pred_reader.read(train_fitem_list)
        train_iter = iterator(train_instances, num_epochs=1, shuffle=True)

        for batch in tqdm(train_iter, desc="Batch Loop"):
            model.train()
            batch = allen_util.move_to_device(batch, device_num)
            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            gt_span = batch['gt_span']

            loss = model(mode=BertSpan.ForwardMode.TRAIN,
                         input_ids=paired_sequence,
                         token_type_ids=paired_segments_ids,
                         attention_mask=att_mask,
                         gt_span=gt_span)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if gradient_accumulate_step > 1:
                loss = loss / gradient_accumulate_step

            loss.backward()
            forbackward_step += 1

            if forbackward_step % gradient_accumulate_step == 0:
                optimizer.step()
                if ema is not None and do_ema:
                    updated_model = model.module if hasattr(model, 'module') else model
                    ema(updated_model.named_parameters())
                optimizer.zero_grad()
                update_step += 1

                if update_step % eval_frequency == 0:
                    # print("Non-EMA EVAL:")
                    # eval_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
                    # cur_eitem_list, cur_eval_dict = span_eval(model, eval_iter, do_lower_case, dev_fitem_dict,
                    #                                           device_num)
                    # cur_results_dict = dict()
                    # cur_results_dict['p_answer'] = cur_eval_dict
                    # cur_results_dict['sp'] = dev_sp_results_dict
                    #
                    # _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
                    # # print(metrics)
                    #
                    # logging_item = {
                    #     'score': metrics,
                    # }
                    #
                    # joint_f1 = metrics['joint_f1']
                    # joint_em = metrics['joint_em']
                    #
                    # print(logging_item)
                    #
                    # if not debug:
                    #     save_file_name = f'i({update_step})|e({epoch_i})' \
                    #         f'|j_f1({joint_f1})|j_em({joint_em})|seed({seed})'
                    #
                    #     # print(save_file_name)
                    #     logging_agent.incorporate_results({}, save_file_name, logging_item)
                    #     logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")
                    #
                    #     model_to_save = model.module if hasattr(model, 'module') else model
                    #     output_model_file = Path(file_path_prefix) / save_file_name
                    #     torch.save(model_to_save.state_dict(), str(output_model_file))

                    if do_ema and ema is not None:
                        print("EMA EVAL")
                        ema_model = ema.get_inference_model()
                        ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=ema_device_num)
                        ema_model = ema_model.to(ema_device_num)
                        ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)
                        dev_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
                        cur_eitem_list, cur_eval_dict = span_eval(ema_model, dev_iter, do_lower_case, dev_fitem_dict,
                                                                  ema_device_num, show_progress=False)
                        cur_results_dict = dict()
                        cur_results_dict['p_answer'] = cur_eval_dict
                        cur_results_dict['sp'] = dev_sp_results_dict

                        _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
                        print(metrics)
                        print("---------------" * 3)

                        logging_item = {
                            'label': 'ema',
                            'score': metrics,
                        }

                        joint_f1 = metrics['joint_f1']
                        joint_em = metrics['joint_em']

                        print(logging_item)

                        if not debug:
                            save_file_name = f'ema_i({update_step})|e({epoch_i})' \
                                f'|j_f1({joint_f1})|j_em({joint_em})|seed({seed})'
                            # print(save_file_name)
                            logging_agent.incorporate_results({}, save_file_name, logging_item)
                            logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

                            model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
                            output_model_file = Path(file_path_prefix) / save_file_name
                            torch.save(model_to_save.state_dict(), str(output_model_file))


if __name__ == '__main__':
    model_go(sent_filter_value=5, sent_top_k=5)
    # model_path = config.PRO_ROOT / "saved_models/05-04-11:28:22_hotpot_v0_qa_(s_top_k:5,s_fv:0.5,qa_layer:2)/ema_i(14000)|e(4)|j_f1(0.4915947771847026)|j_em(0.26603646185010127)|seed(12)"
    # data_files = [
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_1.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_2.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_3.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_4.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_5.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_6.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_7.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_8.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_9.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_10.jsonl",
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_11.jsonl",
    # ]
    # data_files = [
    # data_file = config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_5.jsonl"
    # ]
    # eval_model(model_path, data_file, filter_value=0.5)
    # for d_file in data_files:
    #     eval_model(model_path, d_file)
    # for sent_top_k in [0, 2, 4, 6, 8, 10]:
    # for sent_top_k in [0, 2, 4, 6, 8, 10]:
    #     model_go(sent_filter_value=0, sent_top_k=5)
    # utest_on_squad_v2()
