import os
import sys
import json
import argparse

from tqdm import tqdm


def load_baike_jsonline(file, start_id=0):
    with open(file, 'r', encoding='utf8') as fin:
        for line in tqdm(fin):
            line = line.strip('\r\n')
            if not line:
                continue
            item = json.loads(line)
            question = ''
            answer = ''
            q_keys = ['question', 'title']
            for k in q_keys:
                question = item.get(k, '')
                if question:
                    break
            a_keys = ['answer']
            for k in a_keys:
                answer = item.get(k, '')
                if answer:
                    break
            question = question.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            answer = answer.replace('\r\n', '<br/>').replace('\n', '<br/>').replace('\r', '<br/>').replace('\t', ' ')
            if not question or not answer:
                continue
            category = [cate.strip() for cate in item.get('category', '').split('-')]
            doc = {
                'id': str(start_id),
                'question': question,
                'answer': answer,
                'category': category
            }
            start_id += 1
            yield doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonline", required=True
    )
    parser.add_argument(
        "--output-jsonline", required=True
    )
    parser.add_argument(
        "--start-id", default=0, type=int
    )
    args = parser.parse_args()

    with open(args.output_jsonline, 'w', encoding='utf8') as fout:
        for doc in load_baike_jsonline(args.input_jsonline, args.start_id):
            fout.write('{}\n'.format(json.dumps(doc, ensure_ascii=False)))

    print("Preprocess done!")
