from torch import nn
import torch
import logging

from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SentenceBertClassifier(nn.Module):
    def __init__(self, bert_name_or_dir, num_labels, max_seq_len):
        super(SentenceBertClassifier, self).__init__()

        self.num_labels = num_labels
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name_or_dir)
        self.bert = AutoModel.from_pretrained(bert_name_or_dir)
        self.classifier = nn.Linear(3 * self.bert.config.hidden_size, num_labels)

    def encode(self, sentences):
        device = next(self.parameters()).device
        model_inputs = self.tokenizer.batch_encode_plus(
            sentences,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        masked_output = outputs[0] * attention_mask.unsqueeze(-1)
        pooled_output = masked_output.sum(1) / attention_mask.sum(1).unsqueeze(-1)
        # pooled_output = outputs[1]
        return pooled_output

    def forward(self, sentences_1, sentences_2):
        u = self.encode(sentences_1)
        v = self.encode(sentences_2)
        abs_diff = (u-v).abs()
        features = torch.cat((u, v, abs_diff), dim=-1)
        logits = self.classifier(features)

        return logits


class BertClassifier(nn.Module):

    def __init__(self, bert_name_or_dir, num_labels, max_seq_len):
        super(BertClassifier, self).__init__()

        self.num_labels = num_labels
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name_or_dir)
        self.bert = AutoModel.from_pretrained(bert_name_or_dir)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def encode(self, sentence_pairs):
        device = next(self.parameters()).device
        model_inputs = self.tokenizer.batch_encode_plus(
            sentence_pairs,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        input_ids = model_inputs['input_ids']
        token_type_ids = model_inputs['token_type_ids']
        attention_mask = model_inputs['attention_mask']

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]

        return pooled_output

    def forward(self, sentences_1, sentences_2):
        sentence_pairs = list(zip(sentences_1, sentences_2))
        pooled_output = self.encode(sentence_pairs)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits



