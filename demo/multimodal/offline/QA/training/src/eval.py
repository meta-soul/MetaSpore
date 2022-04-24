import os
import logging
import argparse

from dataset import STSDataset, NLIDataset, QMCDataset

import torch
torch.set_num_threads(3)

from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.ERROR,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model-list", required=True, help="The list of models will be evaluated, name1#model1,name2#model2.")
parser.add_argument("--eval-list", required=True, help="The list of eval data file, name1#file1,name2#file2.")
parser.add_argument("--output-dir", default=None, help="The output path of evaluation results.")
parser.add_argument("--eval-mode", default="sts", choices=["sts"])
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max-seq-len", default=256, type=int)
args = parser.parse_args()

model_list = []
for model_key in args.model_list.split(','):
    model_name, model_name_or_path = model_key.split('#')
    model_list.append([model_name, SentenceTransformer(model_name_or_path, device=args.device)])
    model_list[-1][1].max_seq_length = args.max_seq_len

eval_list = []
for eval_key in args.eval_list.split(','):
    eval_name, eval_file = eval_key.split('#')
    if 'nli' in eval_name:
        dev_samples = NLIDataset(eval_file).load()
        dev_samples = [InputExample(texts=x.texts, label=x.label/3.0) for x in dev_samples]
        #dev_samples = [InputExample(texts=x.texts, label=(1.0 if x.label==2 else 0.0)) for x in dev_samples]
    elif 'qmc' in eval_name:
        dev_samples = QMCDataset(eval_file).load()
        dev_samples = [InputExample(texts=x.texts, label=(1.0 if x.label==1 else 0.0)) for x in dev_samples]
    else:
        dev_samples = STSDataset(eval_file).load()
    dev_evaluator_sts = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, 
        batch_size=args.batch_size, name=eval_name)
    eval_list.append([eval_name, dev_evaluator_sts])

print('Model', 'Evalset', 'Score', sep='\t')
for model_name, model in model_list:
    output_path = None
    if args.output_dir:
        output_path = os.path.join(args.output_dir, model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    for eval_name, evaluator in eval_list:
        score = evaluator(model, output_path=output_path)
        print(model_name, eval_name, score, sep='\t')
