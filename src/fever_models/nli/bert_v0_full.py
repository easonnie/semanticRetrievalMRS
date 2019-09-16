import copy
import os

from allennlp.data.iterators import BasicIterator

from data_utils.exvocab import ExVocabulary
from data_utils.readers.bert_fever_reader import BertReaderFeverNLI
from evaluation import fever_scorer
from fever_sampler.qa_aug_sampler import get_sample_data
from utils import common, save_tool
import config
from utest.bert_mnli_utest.bert_reader import BertReaderMNLI
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification, BertAdam
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

            eval_paired_sequence = eval_paired_sequence.to(device)
            eval_paired_segments_ids = eval_paired_segments_ids.to(device)
            # eval_labels_ids = eval_labels_ids.to(device)
            eval_att_mask = eval_att_mask.to(device)

            out = model(eval_paired_sequence, token_type_ids=eval_paired_segments_ids,
                        attention_mask=eval_att_mask,
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


def eval_model(model, eval_iter, device):
    output_logits_list = []
    output_id_list = []
    output_pred_list = []
    y_pred_list = []
    total_size = 0
    model.eval()

    print(f"Start Eval ({datetime.datetime.now()}):")
    with torch.no_grad():
        for i, batch in enumerate(eval_iter):
            eval_paired_sequence = batch['paired_sequence']
            eval_paired_segments_ids = batch['paired_segments_ids']
            eval_labels_ids = batch['label']
            eval_att_mask, _ = torch_util.get_length_and_mask(eval_paired_sequence)

            eval_paired_sequence = eval_paired_sequence.to(device)
            eval_paired_segments_ids = eval_paired_segments_ids.to(device)
            eval_labels_ids = eval_labels_ids.to(device)
            eval_att_mask = eval_att_mask.to(device)

            eval_logits = model(eval_paired_sequence, token_type_ids=eval_paired_segments_ids,
                                attention_mask=eval_att_mask,
                                labels=None)
            total_size += eval_logits.size(0)

            output_pred_list.extend(torch.max(eval_logits, 1)[1].view(eval_logits.size(0)).tolist())
            output_logits_list.extend(eval_logits.tolist())
            output_id_list.extend(list(batch['pid']))
            y_pred_list.extend(eval_labels_ids.tolist())

    assert len(y_pred_list) == len(output_pred_list)
    correct = 0
    for pred, y in zip(output_pred_list, y_pred_list):
        if pred == y:
            correct += 1

    print(correct, total_size, correct / total_size)


def model_go():
    seed = 6
    bert_model_name = 'bert-base-uncased'
    lazy = False
    forward_size = 16
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 5e-5
    num_train_epochs = 3
    do_ema = False
    dev_prob_threshold = 0.1
    train_prob_threshold = 0.35
    debug_mode = False

    experiment_name = f"bert_fever_nli_baseline_on_fulldata"

    training_file = config.FEVER_TRAIN

    train_sample_top_k = 8

    est_datasize = 208_346

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
        config.FEVER_DEV,
        dev_upstream_sent_list,
        prob_threshold=dev_prob_threshold, top_n=5)

    dev_data_list = fever_nli_sampler.select_sent_with_prob_for_eval(
        config.FEVER_DEV, dev_sent_after_threshold_filter,
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

    # print(list(mnli_dev_instances))

    # Load training model
    # Load training model
    model_clf = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_class)

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

    model_clf.train()

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

        train_data_list = train_data_list

        random.shuffle(train_data_list)
        print("Sample data length:", len(train_data_list))
        sampled_train_instances = bert_fever_reader.read(train_data_list)
        #
        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1)

        for i, batch in enumerate(tqdm(train_iter)):
            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            labels_ids = batch['label']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)

            paired_sequence = paired_sequence.to(device)
            paired_segments_ids = paired_segments_ids.to(device)
            labels_ids = labels_ids.to(device)
            att_mask = att_mask.to(device)

            loss = model_clf(paired_sequence, token_type_ids=paired_segments_ids, attention_mask=att_mask,
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
                                                                                     common.load_jsonl(config.FEVER_DEV),
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
                                                                         common.load_jsonl(config.FEVER_DEV),
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


def model_eval(model_save_path):
    seed = 6
    bert_model_name = 'bert-base-uncased'
    lazy = False
    forward_size = 16
    batch_size = 32
    # dev_prob_threshold = 0.05
    dev_prob_threshold = 0.1

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
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "doc_retri_results/fever_results/sent_results/4-14-sent_results_v0/i(5000)|e(0)|s01(0.9170917091709171)|s05(0.8842384238423843)|seed(12)_dev_sent_results.json")

    # dev_upstream_sent_list = common.load_jsonl(config.DATA_ROOT /
                                               # "utest_data/dev_sent_score_2_shared_task_dev.jsonl")
                                               # "utest_data/dev_sent_score_1_shared_task_dev_docnum(10)_ensembled.jsonl")

    # dev_upstream_sent_list = common.load_jsonl(config.FEVER_DATA_ROOT /
    #                                            "upstream_sentence_selection_Feb16/dev_sent_pred_scores.jsonl")

    dev_upstream_sent_list = common.load_jsonl(config.FEVER_DATA_ROOT /
                                               "upstream_sentence_selection_Feb16/4-15-dev_sent_pred_scores.jsonl")
    #
    # dev_upstream_sent_list = common.load_jsonl(config.FEVER_DATA_ROOT /
    #                                            "upstream_sentence_selection_Feb16/4-15-test_sent_pred_scores.jsonl")

    # dev_upstream_sent_list = common.load_jsonl(config.FEVER_DATA_ROOT /
    #                                            "upstream_sentence_selection_Feb16/n_dev_sent_pred_scores.jsonl")


    # dev_sent_after_threshold_filter = fever_ss_sampler.threshold_sampler_insure_unique_new_format(
    dev_sent_after_threshold_filter = fever_ss_sampler.threshold_sampler_insure_unique(
        config.FEVER_DEV,
        dev_upstream_sent_list,
        prob_threshold=dev_prob_threshold, top_n=5)

    dev_data_list = fever_nli_sampler.select_sent_with_prob_for_eval(
        config.FEVER_DEV, dev_sent_after_threshold_filter,
        None, tokenized=True)

    # dev_sent_after_threshold_filter = fever_ss_sampler.threshold_sampler_insure_unique(
    #     config.FEVER_TEST,
    #     dev_upstream_sent_list,
    #     prob_threshold=dev_prob_threshold, top_n=5)
    #
    # dev_data_list = fever_nli_sampler.select_sent_with_prob_for_eval(
    #     config.FEVER_TEST, dev_sent_after_threshold_filter,
    #     None, tokenized=True, pipeline=True)

    for item in dev_data_list:
        item['label'] = 'hidden'

    dev_list = common.load_jsonl(config.FEVER_DEV)

    for a, b in zip(dev_list, dev_data_list):
        del b['label']
        b['predicted_label'] = a['label']

    eval_mode = {'check_sent_id_correct': True, 'standard': True}
    fever_score, label_score, pr, rec, f1 = fever_scorer.fever_score(dev_data_list,
                                                                     dev_list,
                                                                     mode=eval_mode,
                                                                     verbose=False)
    print("Fever Score(FScore/LScore:/Precision/Recall/F1):", fever_score, label_score, pr, rec, f1)
    print(f"Dev:{fever_score}/{label_score}")

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    bert_fever_reader = BertReaderFeverNLI(bert_tokenizer, lazy=lazy)

    dev_instances = bert_fever_reader.read(dev_data_list)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    # print(list(mnli_dev_instances))

    # Load training model
    # Load training model
    model_clf = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_class)

    model_clf.load_state_dict(torch.load(model_save_path))

    model_clf.to(device)

    model_clf.eval()

    if n_gpu > 1:
        model_clf = nn.DataParallel(model_clf)

    dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

    # for item in dev_data_list:

    dev_data_list = hidden_eval(model_clf, dev_iter, dev_data_list, device)

    common.save_jsonl(dev_data_list, config.PRO_ROOT / "data/fever/upstream_sentence_selection_Feb16/4-15-dev_nli_results.jsonl")

    eval_mode = {'check_sent_id_correct': True, 'standard': True}
    fever_score, label_score, pr, rec, f1 = fever_scorer.fever_score(dev_data_list,
                                                                     common.load_jsonl(config.FEVER_DEV),
                                                                     mode=eval_mode,
                                                                     verbose=False)
    print("Fever Score(FScore/LScore:/Precision/Recall/F1):", fever_score, label_score, pr, rec, f1)

    print(f"Dev:{fever_score}/{label_score}")


if __name__ == '__main__':
    # model_go()
    model_saved_path = config.PRO_ROOT / "saved_models/02-18-19:05:36_bert_fever_nli_baseline_on_fulldata/i(19533)_epoch(2)_dev(0.6744674467446745)_lacc(0.7207720772077207)_seed(6)"
    model_eval(model_saved_path)
    # model_go_pure_aug()