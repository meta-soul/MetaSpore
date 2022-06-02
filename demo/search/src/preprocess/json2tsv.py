import os
import sys
import json

max_len = -1
if len(sys.argv) > 1:
    max_len = int(sys.argv[1])

def process(text, max_len=-1):
    text = text.strip()
    text = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
    if max_len > 0:
        text = text[:max_len]
    return text

for line in sys.stdin:
    item = json.loads(line.strip())
    qid = item['question_id']
    question = process(item['question'], max_len)
    paragraphs = []
    for p in item['answer_paragraphs']:
        pid = p['paragraph_id']
        paragraph = process(p['paragraph_text'], max_len)
        if not paragraph:
            continue
        paragraphs.append([pid, paragraph])

    if not question or not paragraphs:
        continue

    for pid, para in paragraphs:
        print(question, qid, para, pid, sep='\t')
