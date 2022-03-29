import pandas as pd
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from collections import defaultdict

from data_utils import STDDataset
from tqdm import tqdm
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

source_map = {'title': 'title', 'news_url': 'url', 'fact_check_url': 'url', 'kw': 'kw'}

def custom_group(examples, by):
    group2idx = defaultdict(list)
    if by=='source':
        for i, example in enumerate(examples):
            new_source = source_map[example.source]
            group2idx[new_source].append(i)
    elif by=='reply':
        for i, example in enumerate(examples):
            if example.is_reply:
                group2idx['reply'].append(i)
            else:
                group2idx['non_reply'].append(i)
    return group2idx


class STDEvaluator:

    def __init__(self, examples, batch_size, label_lst):

        self.examples = examples
        self.batch_size = batch_size
        self.id2label = {i: label for i, label in enumerate(label_lst)}
        self.label2id = {label: i for i, label in enumerate(label_lst)}

    def compute_metrics(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = sum(y_pred == y_true) / len(y_pred)

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        print(confusion_matrix)

        class_pred = np.sum(confusion_matrix, axis=0)
        true_pred = np.diag(confusion_matrix)
        class_true = np.sum(confusion_matrix, axis=1)

        class_precision = true_pred / class_pred
        class_recall = true_pred / class_true
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)

        class_precision = class_precision.tolist()
        class_recall = class_recall.tolist()
        class_f1 = class_f1.tolist()

        scores = {'accuracy': acc, 'precision': class_precision, 'recall': class_recall, 'f1': class_f1}

        return scores

    def __call__(self, args, model, output_file=None, on=['overall', 'source']):

        dataset = STDDataset(self.examples)
           
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        all_logits = []
        for batch in tqdm(dataloader):
            model.eval()

            del batch['labels']
            with torch.no_grad():
                if args.only_tweet:
                    logits = model(**batch, only_tweet=True)
                else:
                    logits = model(**batch)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)

        probs = F.softmax(all_logits, dim=-1).cpu().numpy()

        y_pred = np.argmax(probs, axis=-1)
        y_true = np.array([self.label2id[example.label] for example in self.examples])

        score_dict = defaultdict(list)
        for type_ in on:
            if type_=='overall':
                scores = self.compute_metrics(y_true, y_pred)
                scores['type']='overall'
                score_dict[type_] = [scores]
            else:
                group2idx = custom_group(self.examples, by=type_)
                for g, gids in group2idx.items():
                    y_true_g = np.array(y_true[gids])
                    y_pred_g = np.array(y_pred[gids])
                    scores = self.compute_metrics(y_true_g, y_pred_g)
                    scores['type'] = g
                    score_dict[type_].append(scores)

        for key, scores_list in score_dict.items():
            metric_strings = [self.id2label[i] for i in range(len(self.id2label))]
            metric_strings = ['acc'] + metric_strings + ['macro']
            print(' & '.join(metric_strings))

            print(f"{key} results")
            for scores in scores_list:
                print(f"Type: {scores['type']}")

                score_strings = [f"{scores['accuracy']*100:.2f}"]
                for i in range(len(self.id2label)):
                    score_strings += [f"{scores['precision'][i]*100:.2f}", f"{scores['recall'][i]*100:.2f}", f"{scores['f1'][i]*100:.2f}"]

                macro_precision = sum(scores['precision'])/len(scores['precision'])
                macro_recall = sum(scores['recall'])/len(scores['recall'])
                macro_f1 = sum(scores['f1'])/len(scores['f1'])
                score_strings += [f"{macro_precision*100:.2f}", f"{macro_recall*100:.2f}", f"{macro_f1*100:.2f}"]
                print(' & '.join(score_strings))
            print('\n')


        if output_file is not None:

            pred_labels = [self.id2label[i] for i in y_pred]
            true_labels = [self.id2label[i] for i in y_true]

            output_df = pd.DataFrame({
                'source': [example.source for example in self.examples],
                'is_reply': [example.is_reply for example in self.examples],
                'sent2_id': [example.sent2_id for example in self.examples],
                'sent2': [example.sent2_orig for example in self.examples],
                'sent1_id': [example.sent1_id for example in self.examples],
                'sent1': [example.sent1_orig for example in self.examples],
                'true': true_labels,
                'pred': pred_labels,
            })
            output_df.to_csv(args.output_file, index=False)
