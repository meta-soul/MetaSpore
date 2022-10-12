import enum
import json
import pandas as pd
import numpy as np
import os

from html import unescape
from sklearn.model_selection import train_test_split

ALLOW_DICT = ({
    'Clothing, Shoes & Jewelry': 'Fashion'
})


def load_file(file_name):
    json_lines = []
    with open(file_name) as fh:
        while line := fh.readline():
            json_lines.append(json.loads(line))
    return json_lines


def preprocess(raw_json_lines):
    result_json_lines = []

    for line in raw_json_lines:
        if 'category' in line and len(line['category']) > 0:
            line['category'][0] = unescape(line['category'][0]).replace('\t', '  ')
            line['title'] = unescape(line['title']).replace('\t', '  ')
            line['label'] = 1 if line['category'][0] in ALLOW_DICT else 0
            result_json_lines.append(line)

    return result_json_lines


def gen_dataset(json_lines, random_state=437):
    json_lines = np.random.choice(json_lines, size=500000, replace=False)
    json_lines = preprocess(json_lines)

    sampled_df = pd.DataFrame.from_dict(json_lines)

    sampled_df = sampled_df[['title', 'category', 'label']]

    train, test = train_test_split(sampled_df, test_size=0.05, random_state=random_state)
    train, val = train_test_split(train, test_size=0.05, random_state=random_state)

    return {'train': train, 'val': val, 'test': test}


def save_dataset(dataset_dict):
    base_path = '/your/working/path/title_to_fashion_500k'
    os.makedirs(base_path, exist_ok=True)

    for name, content in dataset_dict.items():
        content.to_csv(f'{base_path}/{name}.tsv', sep='\t', index=False, header=False)
        print(f'shape of {name} is {content.shape}')


def main():
    print('*** Load file start.')
    json_lines = load_file('/your/working/path/amazon-metadata.category.json')
    print('*** Load file done.')

    print('*** Generate dataset start.')
    dataset_dict = gen_dataset(json_lines)
    print('*** Generate dataset done.')

    print('*** Save dataset start.')
    save_dataset(dataset_dict)
    print('*** Save dataset done.')


if __name__ == '__main__':
    main()
