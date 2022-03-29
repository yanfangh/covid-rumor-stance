from data_utils import read_examples_from_file, STDDataset
import argparse
from model_utils import BertClassifier, SentenceBertClassifier
from evaluator import STDEvaluator
import random
import numpy as np
import os
from trainer import train
import torch
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

def calc_class_weight(examples, label_list):
    class_weight = [0] * len(label_list)
    label2id = {label:i for i, label in enumerate(label_list)}
    for example in examples:
        label_id = label2id[example.label]
        class_weight[label_id] += 1
    max_w = max(class_weight)
    class_weight = [max_w/w for w in class_weight]
    return class_weight


def load_model(args):
    if args.model_type=='bert':
        model = BertClassifier(
            bert_name_or_dir=args.bert_name_or_dir, num_labels=3, max_seq_len=args.max_seq_len)
    elif args.model_type=='sbert':
        model = SentenceBertClassifier(
            bert_name_or_dir=args.bert_name_or_dir, num_labels=3, max_seq_len=args.max_sent_len)

    if args.weight_file is not None:
        state_dict = torch.load(args.weight_file, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

    if args.freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False
    return model

def load_label_lst(file):
    label_lst = []
    with open(file, mode='r') as f:
        for line in f:
            line = line.strip()
            if line=='':
                continue
            label_lst.append(line)
    return label_lst


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--max_sent_len', default=128, type=int)
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--weight_file', default=None, type=str)
    parser.add_argument('--swap_order', action='store_true')

    parser.add_argument('--only_tweet', action='store_true')

    # model parameters
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--bert_name_or_dir', type=str)

    # train parameters
    parser.add_argument('--train_data_names', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--num_train_epochs', default=3, type=int)
    parser.add_argument('--warmup_ratio', default=0.1, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--freeze_bert', action='store_true')


    # loss parameters
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--loss_type', default='ce', type=str)

    # eval parameters
    parser.add_argument('--eval_data_name', type=str)
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--output_file', type=str)

    parser.add_argument('--seed', default=1968, type=int)

    args = parser.parse_args()

    args.device = device

    args.do_train = True if args.train_data_names else False
    args.do_eval, args.eval_during_training = False, False
    if args.eval_data_name:
        if args.do_train:
            args.eval_during_training=True
        else:
            args.do_eval=True


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

    logger.info('Loading model')

    model = load_model(args)
    model.to(args.device)

    if args.do_train:
        train_examples_list = []
        train_label_list = []
        for name in args.train_data_names.split(','):
            train_file = os.path.join(args.data_dir, name, 'samples.json')
            train_examples_list.append(read_examples_from_file(train_file, args.swap_order, args.teacher_pred_file))

        label_file = os.path.join(args.data_dir, name, 'labels_' + args.model_type + '.txt')
        train_label_list = load_label_lst(label_file)

        train_examples = sum(train_examples_list, [])
        print(train_examples[0].sent1_text)
        train_dataset = STDDataset(train_examples)

        class_weight=None
        if args.reweight:
            class_weight = calc_class_weight(train_examples, train_label_list)

        evaluator = None
        if args.eval_during_training:
            eval_file = os.path.join(args.data_dir, args.eval_data_name, 'ours_dev.json')
            label_file = os.path.join(args.data_dir, args.eval_data_name, 'labels_'+args.model_type+'.txt')
            eval_label_list = load_label_lst(label_file)
            eval_examples = read_examples_from_file(eval_file, args.swap_order)
            evaluator = STDEvaluator(eval_examples, batch_size=args.eval_batch_size, label_lst=eval_label_list)

        train(args, model, train_dataset, label_list=train_label_list, evaluator=evaluator,
                class_weight=class_weight)

    if args.do_eval:
        eval_file = os.path.join(args.data_dir, args.eval_data_name, 'ours_test.json')
        label_file = os.path.join(args.data_dir, args.eval_data_name, 'labels_'+args.model_type+'.txt')
        eval_label_list = load_label_lst(label_file)
        eval_examples = read_examples_from_file(eval_file, args.swap_order)
        evaluator = STDEvaluator(eval_examples, batch_size=args.eval_batch_size, label_lst=eval_label_list)
        evaluator(args, model, output_file=args.output_file)














