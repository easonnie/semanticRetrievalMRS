import copy
import os

from allennlp.data.iterators import BasicIterator

from data_utils.exvocab import ExVocabulary
from data_utils.readers.bert_reader_sent_selection import BertReaderSentM
from evaluation import ext_hotpot_eval
from hotpot_fact_selection_sampler.sampler_from_distractor import build_sent_match_data_from_distractor_list, \
    downsample_negative_examples, ID_SEPARATOR
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

import datetime
from tqdm import tqdm

id2label = {
    0: "true",
    1: "false",
    -2: "hidden"
}


def convert_sent_list_to_prediction_file(sent_list):
    pred_dict = {'sp': dict()}

    for sent_item in sent_list:
        sid = sent_item['selection_id']
        oid, title, sent_num = sid.split(ID_SEPARATOR)
        sent_num = int(sent_num)
        # Change this to other later
        if oid not in pred_dict['sp']:
            pred_dict['sp'][oid] = []
        if sent_item['pred_label'] == 'true':
            pred_dict['sp'][oid].append([title, sent_num])

    return pred_dict


def eval_model(model, eval_iter, device, data_item_list):
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
    assert len(y_pred_list) == len(data_item_list)
    correct = 0
    for pred, y in zip(output_pred_list, y_pred_list):
        if pred == y:
            correct += 1
    print(f"Finish Eval ({datetime.datetime.now()}):")
    # print(correct, total_size, correct / total_size)

    for i, item in enumerate(data_item_list):
        assert item['selection_id'] == output_id_list[i]
        item['pred_label'] = id2label[output_pred_list[i]]
        # item

    return data_item_list


def model_go():
    seed = 12
    torch.manual_seed(seed)
    # bert_model_name = 'bert-large-uncased'
    bert_model_name = 'bert-base-uncased'
    experiment_name = 'bert_v0_ss'
    lazy = True
    forward_size = 16
    # batch_size = 64
    batch_size = 128
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5

    do_ema = False
    debug_mode = False

    do_lower_case = True

    # est_datasize = 650_000
    est_datasize = 900_000

    num_class = 2
    # num_train_optimization_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    unk_token_num = {'tokens': 1}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace("true", namespace="labels")
    vocab.add_token_to_namespace("false", namespace="labels")
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')

    # Load Dataset
    train_list = common.load_json(config.TRAIN_FILE)
    dev_list = common.load_json(config.DEV_DISTRACTOR_FILE)
    if debug_mode:
        dev_list = dev_list[:100]

    train_sent_data_list = build_sent_match_data_from_distractor_list(train_list, is_training=True)
    dev_sent_data_list = build_sent_match_data_from_distractor_list(dev_list, is_training=False)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)

    bert_mnli_reader = BertReaderSentM(bert_tokenizer, lazy=lazy)

    dev_instances = bert_mnli_reader.read(dev_sent_data_list)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    # print(list(mnli_dev_instances))

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
    eval_iter_num = 50_000    # Change this to real evaluation.
    best_f1 = 0
    best_em = 0

    for n_epoch in range(num_train_epochs):
        random.shuffle(train_sent_data_list)
        sampled_train_sent_data_list = downsample_negative_examples(train_sent_data_list, 0.1, 1)
        print("Current Sample Size:", len(sampled_train_sent_data_list))
        train_instances = bert_mnli_reader.read(sampled_train_sent_data_list)
        train_iter = biterator(train_instances, num_epochs=1, shuffle=True)

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
                        dev_sent_data_list = eval_model(ema_model_copy, dev_iter, device, dev_sent_data_list)
                    else:
                        dev_sent_data_list = eval_model(model_clf, dev_iter, device, dev_sent_data_list)

                    sp_pred_format = convert_sent_list_to_prediction_file(dev_sent_data_list)
                    score_tracker, metrics = ext_hotpot_eval.eval(sp_pred_format, dev_list)
                    sp_em, sp_f1, sp_prec, sp_recall = metrics['sp_em'], metrics['sp_f1'], metrics['sp_prec'], metrics[
                        'sp_recall']

                    if best_f1 < sp_f1:
                        print("New Best F1")
                        best_f1 = sp_f1

                        save_path = os.path.join(
                            file_path_prefix,
                            f'({update_step})_epoch({n_epoch})_sp_em({sp_em})_sp_f1({sp_f1})_p({sp_prec})_r({sp_recall})_seed({seed})')
                        model_to_save = model_clf.module if hasattr(model_clf,
                                                                    'module') else model_clf
                        output_model_file = os.path.join(file_path_prefix, save_path)
                        torch.save(model_to_save.state_dict(), output_model_file)

                    if best_em < sp_em:
                        print("New Best F1")
                        best_em = sp_em

                        save_path = os.path.join(
                            file_path_prefix,
                            f'({update_step})_epoch({n_epoch})_sp_em({sp_em})_sp_f1({sp_f1})_p({sp_prec})_r({sp_recall})_seed({seed})')
                        model_to_save = model_clf.module if hasattr(model_clf,
                                                                    'module') else model_clf
                        output_model_file = os.path.join(file_path_prefix, save_path)
                        torch.save(model_to_save.state_dict(), output_model_file)

        print("Update steps:", update_step)
        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)
        print("Epoch Evaluation")
        if do_ema and ema_model_copy is not None and ema_tracker is not None:
            print("EMA evaluation.")
            EMA.load_ema_to_model(ema_model_copy, ema_tracker)
            ema_model_copy.to(device)
            if n_gpu > 1:
                ema_model_copy = nn.DataParallel(ema_model_copy)
            dev_sent_data_list = eval_model(ema_model_copy, dev_iter, device, dev_sent_data_list)
        else:
            dev_sent_data_list = eval_model(model_clf, dev_iter, device, dev_sent_data_list)

        sp_pred_format = convert_sent_list_to_prediction_file(dev_sent_data_list)
        score_tracker, metrics = ext_hotpot_eval.eval(sp_pred_format, dev_list)
        sp_em, sp_f1, sp_prec, sp_recall = metrics['sp_em'], metrics['sp_f1'], metrics['sp_prec'], metrics[
            'sp_recall']
        save_path = os.path.join(
            file_path_prefix,
            f'({update_step})_epoch({n_epoch})_sp_em({sp_em})_sp_f1({sp_f1})_p({sp_prec})_r({sp_recall})_seed({seed})')
        model_to_save = model_clf.module if hasattr(model_clf,
                                                    'module') else model_clf
        output_model_file = os.path.join(file_path_prefix, save_path)
        torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == '__main__':
    model_go()

# With scheduler 83.8
