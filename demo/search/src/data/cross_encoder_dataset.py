from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader


class CrossEncoderDataset(Dataset):

    def __init__(self, data_file, text_indices=[0, 2], label_index=-1, label_converter=int):
        """
        :param data_file: the tsv file with 4 fields ('query', 'title', 'paragraph', 'label')
        """
        assert isinstance(text_indices, (list, tuple)) and (len(text_indices) == 1 or len(text_indices) == 2)
        self.examples = []
        self.text_indices = text_indices
        Example = namedtuple('Example', ['texts', 'label'])
        with open(data_file, 'r', encoding='utf8') as fin:
            for line in fin:
                #['query', 'title', 'paragraph', 'label']
                row = line.rstrip('\n').split('\t')
                texts = []
                for i in text_indices:
                    row[i] = row[i].replace(' ', '')  # remove space in the text
                    texts.append(row[i])
                self.examples.append(Example(texts, label_converter(row[label_index])))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class CrossEncoderCollater(object):

    def __init__(self, tokenizer, device=None):
        self.tokenizer = tokenizer
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

        features = self.tokenizer(texts_a, texts_b)

        if isinstance(labels[0], int):
            labels = torch.tensor(labels).long()
        else:
            labels = torch.tensor(labels).float()

        if self.device is not None:
            features = {k:v.to(self.device) for k, v in features.items()}
            labels = labels.to(self.device)

        return features, labels


def create_cross_encoder_dataset(data_file, text_indices=[0, 2], label_index=-1, label_converter=int):
    dataset = CrossEncoderDataset(data_file, text_indices=text_indices,
        label_index=label_index, label_converter=label_converter)
    return dataset


def create_cross_encoder_dataloader(data_file, tokenizer, text_indices=[0, 2], label_index=-1, label_converter=int,
        batch_size=64, shuffle=True, device=None, num_workers=0, pin_memory=False, drop_last=False):
    collater = CrossEncoderCollater(tokenizer, device)
    dataset = create_cross_encoder_dataset(data_file, text_indices, label_index, label_converter)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collater, pin_memory=pin_memory, 
        drop_last=drop_last)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    data_file = '../../data/dev/dev.q.format'

    def tokenize(text_a, text_b):
        return tokenizer(text_a, text_b, return_tensors='pt', padding=True, truncation=True)

    dataloader = create_cross_encoder_dataloader(data_file, tokenize, text_indices=[0], device='cuda:0')
    #dataloader = create_cross_encoder_dataloader(data_file, tokenize, text_indices=[0, 1], device='cuda:0')
    for features, labels in dataloader:
        print('features', features)
        print('labels', labels)
