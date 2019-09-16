import copy
import os
from enum import Enum

from allennlp.data.iterators import BasicIterator
from pytorch_pretrained_bert.modeling import BertLayerNorm

from bert_model_variances.bert_multilayer_output import BertMultiLayerSeqClassification
from data_utils.exvocab import ExVocabulary
from data_utils.readers.bert_fever_reader import BertReaderFeverNLI
from evaluation import fever_scorer
from fever_sampler.qa_aug_sampler import get_sample_data
from utils import common, save_tool
import config
from utest.bert_mnli_utest.bert_reader import BertReaderMNLI
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from flint import torch_util
import torch
import torch.nn as nn
import random
import torch.optim as optim
from neural_modules.model_EMA import EMA
import fever_sampler.sentence_selection_sampler as fever_ss_sampler
import fever_sampler.nli_sampler as fever_nli_sampler
import torch.nn.functional as F

import datetime
from tqdm import tqdm


def hidden_eval(model, data_iter, dev_data_list, device, with_logits=False, with_probs=False):
    # SUPPORTS < (-.-) > 0
    # REFUTES < (-.-) > 1
    # NOT ENOUGH INFO < (-.-) > 2

    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }

    print("Evaluating ...")
    with torch.no_grad():
        model.eval()
        totoal_size = 0

        y_pred_list = []
        y_id_list = []

        y_logits_list = []
        y_probs_list = []

        # if append_text:
        # y_premise = []
        # y_hypothesis = []

        for batch_idx, batch in enumerate(data_iter):

            eval_paired_sequence = batch['paired_sequence']
            eval_paired_segments_ids = batch['paired_segments_ids']
            # eval_labels_ids = batch['label']
            eval_att_mask, _ = torch_util.get_length_and_mask(eval_paired_sequence)
            eval_s1_span, eval_s2_span = batch['bert_s1_span'], batch['bert_s2_span']

            eval_paired_sequence = eval_paired_sequence.to(device)
            eval_paired_segments_ids = eval_paired_segments_ids.to(device)
            # eval_labels_ids = eval_labels_ids.to(device)
            eval_att_mask = eval_att_mask.to(device)
            eval_s1_span = eval_s1_span.to(device)
            eval_s2_span = eval_s2_span.to(device)

            out = model(eval_paired_sequence, token_type_ids=eval_paired_segments_ids,
                        attention_mask=eval_att_mask,
                        mode=BertMultiLayerSeqClassification.ForwardMode.EVAL,
                        labels=None)

            y_id_list.extend(list(batch['pid']))

            # if append_text:
            # y_premise.extend(list(batch['text']))
            # y_hypothesis.extend(list(batch['query']))

            y_pred_list.extend(torch.max(out, 1)[1].view(out.size(0)).tolist())

            if with_logits:
                y_logits_list.extend(out.tolist())

            if with_probs:
                y_probs_list.extend(F.softmax(out, dim=1).tolist())

            totoal_size += out.size(0)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_pred_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['id'])

            # Matching id
            dev_data_list[i]['predicted_label'] = id2label[y_pred_list[i]]
            if with_logits:
                dev_data_list[i]['logits'] = y_logits_list[i]

            if with_probs:
                dev_data_list[i]['probs'] = y_probs_list[i]

            # Reset neural set
            if len(dev_data_list[i]['predicted_sentids']) == 0:
                dev_data_list[i]['predicted_label'] = "NOT ENOUGH INFO"

            # if append_text:
            #     dev_data_list[i]['premise'] = y_premise[i]
            #     dev_data_list[i]['hypothesis'] = y_hypothesis[i]

        print('total_size:', totoal_size)

    return dev_data_list


def model_go():
    # bert_model_name = 'bert-large-uncased'
    seed = 6
    bert_model_name = 'bert-base-uncased'
    lazy = False
    forward_size = 16
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5
    do_ema = False
    dev_prob_threshold = 0.1
    train_prob_threshold = 0.35
    debug_mode = False
    experiment_name = f"fever_nli_bert_multilayer_matching_pretrained_pooler"
    num_of_pooling_layer = 1
    use_pretrained_pooler = True

    data_aug = False
    data_aug_file = config.FEVER_DATA_ROOT / "qa_aug/squad_train_turker_groundtruth.json"
    # data_aug_size = int(21_015 * some_params)   # 10p
    # data_aug_size = int(208_346 * 0)
    data_aug_size = int(0)

    # training_file = config.FEVER_DATA_ROOT / "fever_1.0/train_10.jsonl"
    training_file = config.FEVER_DATA_ROOT / "fever_1.0/train.jsonl"

    train_sample_top_k = 8

    # est_datasize = 208_346    # full
    # est_datasize = 14_544
    # est_datasize = 21_015 + data_aug_size   # 10p
    est_datasize = 208_346 + data_aug_size

    num_class = 3

    # num_train_optimization_steps
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    unk_token_num = {'tokens': 1}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace('SUPPORTS', namespace='labels')
    vocab.add_token_to_namespace('REFUTES', namespace='labels')
    vocab.add_token_to_namespace('NOT ENOUGH INFO', namespace='labels')
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')
    # Finished build vocabulary.

    # Load standardized sentence file
    dev_upstream_sent_list = common.load_jsonl(config.FEVER_DATA_ROOT /
                                               "upstream_sentence_selection_Feb16/dev_sent_pred_scores.jsonl")
    dev_sent_after_threshold_filter = fever_ss_sampler.threshold_sampler_insure_unique(
        config.FEVER_DATA_ROOT / "fever_1.0/shared_task_dev.jsonl",
        dev_upstream_sent_list,
        prob_threshold=dev_prob_threshold, top_n=5)

    dev_data_list = fever_nli_sampler.select_sent_with_prob_for_eval(
        config.FEVER_DATA_ROOT / "fever_1.0/shared_task_dev.jsonl", dev_sent_after_threshold_filter,
        None, tokenized=True)

    # print(dev_data_list[0])
    # exit(0)

    train_upstream_sent_list = common.load_jsonl(config.FEVER_DATA_ROOT /
                                                 "upstream_sentence_selection_Feb16/train_sent_scores.jsonl")
    # Finished loading standardized sentence file.

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    bert_fever_reader = BertReaderFeverNLI(bert_tokenizer, lazy=lazy)

    dev_instances = bert_fever_reader.read(dev_data_list)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    # Load training model
    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model_clf = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class,
                                                act_type='tanh',
                                                num_of_pooling_layer=num_of_pooling_layer,
                                                use_pretrained_pooler=use_pretrained_pooler)

    ema_tracker = None
    ema_model_copy = None
    if do_ema and ema_tracker is None:
        ema_tracker = EMA(model_clf.named_parameters(), on_cpu=True)
        ema_model_copy = copy.deepcopy(model_clf)

    model_clf.to(device)

    param_optimizer = list(model_clf.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(est_datasize / forward_size / gradient_accumulate_step) * \
                                   num_train_epochs

    print(num_train_optimization_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_optimization_steps)

    # optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate)

    # # # Create Log File
    file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # # # Log File end

    if n_gpu > 1:
        model_clf = nn.DataParallel(model_clf)

    forbackward_step = 0
    update_step = 0
    eval_iter_num = 2_000  # Change this to real evaluation.
    best_fever_score = -1

    for n_epoch in range(num_train_epochs):
        print("Resampling...")
        train_sent_after_threshold_filter = \
            fever_ss_sampler.threshold_sampler_insure_unique(training_file,
                                                             train_upstream_sent_list,
                                                             train_prob_threshold,
                                                             top_n=train_sample_top_k)
        #
        train_data_list = fever_nli_sampler.adv_simi_sample_with_prob_v1_1(
            training_file,
            train_sent_after_threshold_filter,
            None,
            tokenized=True)

        aug_d_list = []
        if data_aug:
            aug_d_list = common.load_json(data_aug_file)
            random.shuffle(aug_d_list)
            aug_d_list = aug_d_list[:data_aug_size]

        train_data_list = train_data_list + aug_d_list

        random.shuffle(train_data_list)
        print("Sample data length:", len(train_data_list))
        sampled_train_instances = bert_fever_reader.read(train_data_list)
        #
        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1)

        for i, batch in enumerate(tqdm(train_iter)):
            model_clf.train()
            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            labels_ids = batch['label']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            s1_span = batch['bert_s1_span']
            s2_span = batch['bert_s2_span']

            paired_sequence = paired_sequence.to(device)
            paired_segments_ids = paired_segments_ids.to(device)
            labels_ids = labels_ids.to(device)
            att_mask = att_mask.to(device)
            s1_span = s1_span.to(device)
            s2_span = s2_span.to(device)

            loss = model_clf(paired_sequence, token_type_ids=paired_segments_ids, attention_mask=att_mask,
                             mode=BertMultiLayerSeqClassification.ForwardMode.TRAIN,
                             labels=labels_ids)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if gradient_accumulate_step > 1:
                loss = loss / gradient_accumulate_step

            loss.backward()
            forbackward_step += 1

            if forbackward_step % gradient_accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_step += 1
                if do_ema and ema_tracker is not None:
                    # if model_clf is DataParallel, then we use model_clf.module
                    model_to_track = model_clf.module if hasattr(model_clf,
                                                                 'module') else model_clf
                    ema_tracker(model_to_track.named_parameters())  # Whenever we do update, the do ema update

                if update_step % eval_iter_num == 0:
                    print("Update steps:", update_step)
                    dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

                    if do_ema and ema_model_copy is not None and ema_tracker is not None:
                        print("EMA evaluation.")
                        EMA.load_ema_to_model(ema_model_copy, ema_tracker)
                        ema_model_copy.to(device)
                        if n_gpu > 1:
                            ema_model_copy = nn.DataParallel(ema_model_copy)
                        dev_data_list = hidden_eval(ema_model_copy, dev_iter, dev_data_list, device)
                    else:
                        dev_data_list = hidden_eval(model_clf, dev_iter, dev_data_list, device)

                    eval_mode = {'check_sent_id_correct': True, 'standard': True}
                    fever_score, label_score, pr, rec, f1 = fever_scorer.fever_score(dev_data_list,
                                                                                     common.load_jsonl(config.FEVER_DATA_ROOT / "fever_1.0/shared_task_dev.jsonl"),
                                                                                     mode=eval_mode,
                                                                                     verbose=False)
                    print("Fever Score(FScore/LScore:/Precision/Recall/F1):", fever_score, label_score, pr, rec, f1)

                    print(f"Dev:{fever_score}/{label_score}")

                    if best_fever_score < fever_score:
                        print("New Best FScore")
                        best_fever_score = fever_score

                        save_path = os.path.join(
                            file_path_prefix,
                            f'i({update_step})_epoch({n_epoch})_dev({fever_score})_lacc({label_score})_seed({seed})'
                        )
                        model_to_save = model_clf.module if hasattr(model_clf,
                                                                    'module') else model_clf
                        output_model_file = os.path.join(file_path_prefix, save_path)
                        torch.save(model_to_save.state_dict(), output_model_file)

        print("Update steps:", update_step)
        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

        if do_ema and ema_model_copy is not None and ema_tracker is not None:
            print("EMA evaluation.")
            EMA.load_ema_to_model(ema_model_copy, ema_tracker)
            ema_model_copy.to(device)
            if n_gpu > 1:
                ema_model_copy = nn.DataParallel(ema_model_copy)
            dev_data_list = hidden_eval(ema_model_copy, dev_iter, dev_data_list, device)
        else:
            dev_data_list = hidden_eval(model_clf, dev_iter, dev_data_list, device)

        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        fever_score, label_score, pr, rec, f1 = fever_scorer.fever_score(dev_data_list,
                                                                         common.load_jsonl(config.FEVER_DATA_ROOT / "fever_1.0/shared_task_dev.jsonl"),
                                                                         mode=eval_mode,
                                                                         verbose=False)
        print("Fever Score(FScore/LScore:/Precision/Recall/F1):", fever_score, label_score, pr, rec, f1)

        print(f"Dev:{fever_score}/{label_score}")

        if best_fever_score < fever_score:
            print("New Best FScore")
            best_fever_score = fever_score

            save_path = os.path.join(
                file_path_prefix,
                f'i({update_step})_epoch({n_epoch})_dev({fever_score})_lacc({label_score})_seed({seed})'
            )
            model_to_save = model_clf.module if hasattr(model_clf,
                                                        'module') else model_clf
            output_model_file = os.path.join(file_path_prefix, save_path)
            torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == '__main__':
    model_go()