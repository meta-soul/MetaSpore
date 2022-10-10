from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader


class TextClassificationDataset(Dataset):

    def __init__(self, data_file, text_indices=[0], label_index=None, label_converter=int):
        assert isinstance(text_indices, (list, tuple)) and (len(text_indices) == 1 or len(text_indices) == 2)
        self.examples = []
        self.text_indices = text_indices

        Example = namedtuple('Example', ['texts', 'label'])
        with open(data_file, 'r', encoding='utf8') as fin:
            for line in fin:
                row = line.rstrip('\r\n').split('\t')
                texts = [row[i] for i in text_indices]
                if label_index is None:
                    label = None  # no-label, should be inference data
                elif ',' in row[label_index]:
                    label = [label_converter(l) for l in row[label_index].split(',')]  # multi-label
                else:
                    label = label_converter(row[label_index])  # single-label

                self.examples.append(Example(texts, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TextClassificationDataCollater(object):

    def __init__(self, preprocess, device=None):
        self.preprocess = preprocess
        self.device = device

    def __call__(self, batch):
        texts_a, texts_b, labels = [], [], []

        labels = [e.label for e in batch]

        num_texts = len(batch[0].texts)
        if num_texts == 1:
            texts_a = [e.texts[0] for e in batch]
            texts_b = None
        else:
            texts_a = [e.texts[0] for e in batch]
            texts_b = [e.texts[1] for e in batch]

        features = self.preprocess(texts_a, texts_b)

        if labels[0] is None:
            labels = None
        elif isinstance(labels[0], int):
            labels = torch.tensor(labels).long()
        else:
            labels = torch.tensor(labels).float()

        if self.device is not None:
            features = {k:v.to(self.device) for k, v in features.items()}
            labels = None if labels is None else labels.to(self.device)

        return features, labels


def create_text_classification_dataset(data_file, text_indices=[0], label_index=1, label_converter=int):
    dataset = TextClassificationDataset(data_file, text_indices=text_indices,
        label_index=label_index, label_converter=label_converter)
    return dataset


def create_text_classification_dataloader(data_file, preprocess, 
        text_indices=[0], label_index=1, label_converter=int,
        batch_size=64, shuffle=False, device=None, num_workers=0, 
        pin_memory=False, drop_last=False):
    collater = TextClassificationDataCollater(preprocess, device)
    dataset = create_text_classification_dataset(data_file, text_indices, label_index, label_converter)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collater, pin_memory=pin_memory, 
        drop_last=drop_last)

def test_text_classification_dataset(data_file, text_indices=[0], label_index=1, label_converter=int):
    dataset = create_text_classification_dataset(data_file, text_indices, label_index, label_converter)
    for i, example in enumerate(dataset):
        print(i, example)
        if i >= 5:
            break

def test_text_classification_dataloader(data_file, preprocess, text_indices=[0], label_index=1, label_converter=int, device='cuda:0'):
    dataloader = create_text_classification_dataloader(data_file, preprocess, text_indices, label_index, label_converter,
        device=device, batch_size=16)
    for batch in dataloader:
        features, labels = batch
        print('features', features)
        print('labels', labels, labels.size())
        break
