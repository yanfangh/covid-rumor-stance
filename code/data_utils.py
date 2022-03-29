from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset
import json

from text_preprocess import *
from text_preprocess_args import text_preprocess_args

@dataclass
class STDExample:
    sent1_id: str
    sent1_orig: str
    sent1_text: str
    sent2_id: str
    sent2_orig: str
    sent2_text: str
    label: Optional[str]=None
    source: Optional[str]=None
    is_reply: Optional[str]=None


class STDDataset(Dataset):

    def __init__(self, examples):
        self.num = len(examples)
        self.sentences_1 = [example.sent1_text for example in examples]
        self.sentences_2 = [example.sent2_text for example in examples]

        self.labels = None
        if examples[0].label is not None:
            self.labels = [example.label for example in examples]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):

        item = {
            'sentences_1': self.sentences_1[idx],
            'sentences_2': self.sentences_2[idx],
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


def read_examples_from_file(file, swap_order=False):

    examples = []

    with open(file, mode='r') as f:
        for line in f:
            line = json.loads(line)
            sent1_id = str(line['sent1_id'])
            sent1_orig = line['sentence1']
            sent1_text = preprocess_bert(sent1_orig, text_preprocess_args)
            # sent1_text = sent1_orig
            sent2_id = str(line['sent2_id'])
            sent2_orig = line['sentence2']
            sent2_text = preprocess_bert(sent2_orig, text_preprocess_args)
            # sent2_text = sent2_orig

            label = line['label'] if 'label' in line else None
            if label is None:
                print(1)
            source = line['source'] if 'source' in line else None
            is_reply = line['is_reply'] if 'is_reply' in line else None

            if swap_order:
                sent1_id, sent2_id = sent2_id, sent1_id
                sent1_orig, sent2_orig = sent2_orig, sent1_orig
                sent1_text, sent2_text = sent2_text, sent1_text

            examples.append(STDExample(
                sent1_id=sent1_id,
                sent1_orig=sent1_orig,
                sent1_text=sent1_text,
                sent2_id=sent2_id,
                sent2_orig=sent2_orig,
                sent2_text=sent2_text,
                label=label,
                source=source,
                is_reply=is_reply,
            ))
    logger.info(f"{len(examples)} examples")
    return examples








