import os
import json
import numpy as np

import torch
# torch.set_num_threads(4)
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig


class TextClassificationModel(nn.Module):

    def __init__(self, model_name_or_path, task_type='multiclass', 
            num_labels=15, dropout_p=0.1, device=None, max_seq_len=128, 
            do_lower_case=False):
        super().__init__()
        assert task_type in ['multiclass', 'multilabel', 'binary_classification']
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        if task_type == "multilabel":
            self.loss_fct = nn.BCEWithLogitsLoss()
            self.act_fct = nn.Sigmoid()
        elif task_type == "multiclass":
            self.loss_fct = nn.CrossEntropyLoss()
            self.act_fct = nn.Softmax(dim=1)
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()
            self.process_label = lambda x: torch.unsqueeze(x, -1).float()
            self.act_fct = nn.Sigmoid()

        if device is not None:
            self.to(device)

        self.num_labels = num_labels
        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len
        self.do_lower_case = do_lower_case
        self.task_type = task_type
        self.input_names = self.tokenizer.model_input_names

    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, 'model.pt')
        torch.save(self.state_dict(), model_path)

        conf_path = os.path.join(save_path, 'model.config.json')
        with open(conf_path, 'w', encoding='utf8') as f:
            json.dump({
                'num_labels': self.num_labels,
                'max_seq_len': self.max_seq_len,
                'do_lower_case': self.do_lower_case,
                'task_type': self.task_type,
                'dropout_p': self.dropout_p
            }, f, indent=4)

        self.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)

    @classmethod
    def load(cls, save_path, *args, **kwargs):
        conf_path = os.path.join(save_path, 'model.config.json')
        conf = {}
        with open(conf_path, 'r') as f:
            conf.update(json.load(f))
        conf.update(kwargs)
        model = cls(save_path, *args, **conf)
        model_path = os.path.join(save_path, 'model.pt')
        model.load_state_dict(torch.load(model_path))
        model.to(torch.device("cuda"))
        model.eval()
        return model

    def preprocess(self, text, text_pair=None, padding=True, truncation=True, 
            add_special_tokens=True, return_tensors="pt", device=None):
        if isinstance(text, str):
            text = [text]
        if isinstance(text_pair, str):
            text_pair = [text_pair]
        if self.do_lower_case:
            text = [s.lower() for s in text]
        features = self.tokenizer(text, text_pair=text_pair, padding=padding, truncation=truncation, 
            add_special_tokens=add_special_tokens, return_tensors=return_tensors, 
            max_length=self.max_seq_len)
        if device is not None:
            features = {k:v.to(device) for k, v in features.items()}
        return features

    def forward(self, input_ids: Tensor=None, token_type_ids: Tensor=None, attention_mask: Tensor=None, 
            positions_ids: Tensor=None, labels: Tensor=None, *args, **kwargs):
        inputs = {}
        if 'input_ids' in self.input_names:
            inputs['input_ids'] = input_ids
        if 'attention_mask' in self.input_names:
            inputs['attention_mask'] = attention_mask
        if 'token_type_ids' in self.input_names:
            inputs['token_type_ids'] = token_type_ids
        if 'positions_ids' in self.input_names:
            inputs['positions_ids'] = positions_ids

        outputs = self.model(**inputs)
        cls_pooled = self.dropout(outputs[1])
        logits = self.classifier(cls_pooled)

        if labels is None:
            return logits

        # labels = torch.nn.functional.one_hot(labels, num_classes=self.config.num_labels).float()
        labels = self.process_label(labels)
        # print("******", logits.shape, labels.shape)
        loss = self.loss_fct(logits, labels)
        return logits, loss

    def predict(self, sentences, batch_size=32, convert_to_numpy=True, 
            convert_to_tensor=False, device=None, **kwargs):
        if device is None:
            device = next(self.parameters()).device
            

        def collate_fn(batch):
            if isinstance(batch[0], list):
                texts_a = [x[0] for x in batch]
                texts_b = [x[1] for x in batch]
            else:
                texts_a = batch 
                texts_b = None
            features = self.preprocess(texts_a, texts_b)
            features = {k:v.to(device) for k, v in features.items()}
            return features

        data_loader = DataLoader(sentences, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=0, shuffle=False)

        pred_scores = []
        with torch.no_grad():
            # for features in data_loader:
            for _, features in enumerate(tqdm(data_loader)):
                logits = self.forward(**features)
                scores = self.act_fct(logits)
                pred_scores.extend(scores)
        
        if self.num_labels == 1:
            pred_scores = [scores[0] for scores in pred_scores]  # only has one output
        elif self.num_labels == 2 and self.task_type == "multiclass":
            pred_scores = [scores[1] for scores in pred_scores]  # only return positive score

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        return pred_scores


if __name__ == '__main__':
    
    import pandas as pd
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    test_df = pd.read_csv("/your/working/path/title_to_fashion_500k/test.tsv", sep='\t', header=None)
    test_df = test_df.dropna()
    sentences = test_df[0].to_list()
    labels = test_df[2].to_numpy()

    model = TextClassificationModel.load('/your/working/path/output/title_to_fashion/20220831_173000/epoch_0')
    scores = model.predict(sentences)

    test_df['score'] = scores.tolist()

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    threshold = None
    for p, r, t in zip(precision, recall, thresholds):
        if (r <= 0.91):
            threshold = t
            print("precision: ", p, " recall: ", r, " thresholds: ", t)
            break     

    predictions = scores > threshold
    print('Accuracy: ', (predictions == labels).sum() / len(labels))
    print('AUC: ', roc_auc_score(labels, scores))
    print('F1 Score: ', f1_score(labels, predictions))
