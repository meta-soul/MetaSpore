#
# Copyright 2022 DMetaSoul
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import json
import time
import random
import argparse
import logging
from datetime import datetime

import torch
from torch import nn
import transformers
#from transformers import AdamW
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

from data import create_cross_encoder_dataloader
from modeling import TransformerCrossEncoder

def setup_logger(log_file, log_level, name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    file_handler = logging.FileHandler(filename=log_file)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    stream_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def get_scheduler(name, optimizer, warmup_steps, total_steps):
    if name == "constant":
        return transformers.get_constant_schedule(optimizer)
    elif name == "constant_with_warmup":
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif name == "linear_with_warmup":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    elif name == "cosine_with_warmup":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    raise ValueError(f"Not support scheduler: {name}")

def get_evaluator(args, mrr_at_k=10):
    eval_queries, eval_corpus, eval_rel, eval_irrel = {}, {}, {}, {}
    for qid, query in load_tsv(args.eval_qid_file):
        eval_queries[qid] = query
    for pid, para in load_tsv(args.eval_pid_file):
        eval_corpus[pid] = para
    for qid, pids in load_tsv(args.eval_rel_file):
        pids = pids.split(',')
        eval_rel[qid] = set(pids)
    if os.path.isfile(args.eval_irrel_file):
        for qid, pids in load_tsv(args.eval_irrel_file):
            pids = pids.split(',')
            if qid in eval_rel:
                eval_irrel[qid] = pids - eval_rel[qid]
            else:
                eval_irrel[qid] = pids
    else:
        all_pids = list(eval_corpus.keys())
        for qid in eval_rel:
            eval_irrel[qid] = set(random.choices(all_pids, k=len(eval_rel[qid]))) - eval_rel[qid]

    samples = []
    for qid, query in eval_queries.items():
        if qid not in eval_rel:
            continue
        if qid not in eval_irrel:
            continue
        x = {'query': query, 'positive': [], 'negative': []}
        for pid in eval_rel[qid]:
            if pid not in eval_corpus:
                continue
            x['positive'].append(eval_corpus[pid])
        for pid in eval_irrel[qid]:
            if pid not in eval_corpus:
                continue
            x['negative'].append(eval_corpus[pid])
        if len(x['positive']) == 0 or len(x['negative']) == 0:
            continue
        samples.append(x)

    evaluator = CERerankingEvaluator(samples, mrr_at_k=mrr_at_k, write_csv=False)
    return evaluator, samples

def load_tsv(file):
    with open(file, 'r', encoding='utf8') as fin:
        for line in fin:
            yield line.strip().split('\t')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='The name of experiment'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help="The train data file."
    )
    parser.add_argument(
        '--train-kind',
        type=str,
        default='multiclass',
        choices=['multiclass', 'multilabel', 'regression'],
        help='The train data mode.'
    )
    parser.add_argument(
        '--train-text-index',
        type=str,
        default='0,1',
        help='The index of text fields in the train data.'
    )
    parser.add_argument(
        '--train-label-index',
        type=int,
        default=2,
        help='The index of label fields in the train data.'
    )
    parser.add_argument(
        '--train-label-type',
        type=str,
        default='int',
        help='The label dtype'
    )
    parser.add_argument(
        '--num-labels',
        type=int,
        default=2,
        help='The number of class labels.'
    )
    parser.add_argument(
        '--eval-qid-file',
        type=str,
        default='',
        help='The query id to text file, with format: qid\tquery'
    )
    parser.add_argument(
        '--eval-pid-file',
        type=str,
        default='',
        help='The passage id to text file, with format: pid\tpassage'
    )
    parser.add_argument(
        '--eval-rel-file',
        type=str,
        default='',
        help='The relevant relation between qid and pid, with format: qid\tpid1,pid2,pid3'
    )
    parser.add_argument(
        '--eval-irrel-file',
        type=str,
        default='',
        help='The irrelevant relation between qid and pid, with format: qid\tpid1,pid2,pid3'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='DMetaSoul/sbert-chinese-general-v2',
        help='The pretrained model for training.'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine_with_warmup',
        choices=['constant', 'constant_with_warmup', 'linear_with_warmup', 'cosine_with_warmup']
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9
    )
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-08
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='The max gradient norm for clip.'
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--save-steps',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--use-amp',
        action='store_true'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./output'
    )
    parser.add_argument(
        '--tensorboard-path',
        type=str,
        default=''
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=''
    )
    parser.add_argument(
        '--debug',
        action='store_true'
    )
    return parser.parse_args()

def train(model, train_loader, evaluator, optimizer, scheduler, scaler, summary, logger, args):
    model.train()
    model.zero_grad()
    skip_scheduler = False
    train_steps = len(train_loader) * args.epoch
    for i, (features, labels) in enumerate(tqdm(train_loader)):
        train_steps += 1

        features = {k:v.cuda(args.gpu, non_blocking=True) for k,v in features.items()}
        if labels is not None:
            labels = labels.cuda(args.gpu, non_blocking=True)

        if args.use_amp:
            with autocase():
                _, loss = model(**features, labels=labels)
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            skip_scheduler = scaler.get_scale() != scale_before_step
        else:
            _, loss = model(**features, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad()
        if not skip_scheduler:
            scheduler.step()

        logger.info(
            f"Train Epoch: {args.epoch}\tSteps: {train_steps}\t"
            f"Loss: {loss.item():.6f}\tLR: {optimizer.param_groups[0]['lr']:7f}\t"
        )

        if summary is not None:
            summary.add_scalar("loss", loss.item(), train_steps)
            summary.add_scalar("lr", optimizer.param_groups[0]["lr"], train_steps)

        if args.do_eval and args.eval_steps > 0 and train_steps % args.eval_steps == 0:
            evaluate(model, evaluator, summary, train_steps, logger, args)

        if args.save_steps > 0 and train_steps % args.save_steps == 0:
            save_path = os.path.join(args.output_path, f"step_{train_steps}")
            model.save(save_path)
            evaluate(model, evaluator, summary, train_steps, logger, args,
                save_file=os.path.join(save_path, 'metrics.json'))

        if args.debug:
            break

def evaluate(model, evaluator, summary, steps, logger, args, save_file=None):
    if not args.do_eval or evaluator is None:
        return
    model.eval()
    with torch.no_grad():
        mrr = evaluator(model)
    logger.info(
        f"Eval Epoch: {args.epoch}\tSteps: {steps}\tMRR: {mrr}"
    )
    if summary is not None:
        summary.add_scalar("mrr", mrr, steps)
    if save_file is not None:
        with open(save_file, 'w', encoding='utf8') as f:
            f.write(json.dumps({
                "mrr": mrr
            }, ensure_ascii=False))
    return mrr

def main():
    args = parse_args()

    args.output_path = os.path.join(args.output_path, 
        args.name,
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    os.makedirs(args.output_path, exist_ok=True)
    if not args.log_file:
        args.log_file = os.path.join(args.output_path, 'logs.txt')
    if not args.tensorboard_path:
        args.tensorboard_path = os.path.join(args.output_path, 'runs')
    args.log_level = logging.DEBUG if args.debug else logging.INFO

    if all([os.path.isfile(f) for f in [args.eval_qid_file, args.eval_pid_file, args.eval_rel_file]]):
        args.do_eval = True
    else:
        args.do_eval = False

    # debug just for testing code
    if args.debug:
        args.num_epochs = 1
        args.save_steps = 1
        args.eval_steps = 1
        args.train_batch_size = 16
        args.eval_batch_size = 16

    # logger
    logger = setup_logger(args.log_file, args.log_level)

    # model
    model = TransformerCrossEncoder.load_pretrained(args.model,
        num_labels=args.num_labels, task_type=args.train_kind)
    model.cuda(args.gpu)

    # train data
    label_index = int(args.train_label_index)
    label_converter = float if args.train_label_type == 'float' else int
    text_indices = [int(i) for i in args.train_text_index.split(',')]
    train_loader = create_cross_encoder_dataloader(args.train_file, model.tokenize, 
        text_indices=text_indices, label_index=label_index, label_converter=label_converter,
        batch_size=args.train_batch_size, shuffle=True)
    logger.info(f"Train data batches: {len(train_loader)}")

    #for features, labels in train_loader:
    #    print(features, labels)
    #    break

    # eval data
    evaluator = None
    if args.do_eval:
        evaluator, _ = get_evaluator(args, mrr_at_k=10)
        logger.info(f"Evaluate with {len(_)} queries.")
    else:
        logger.info("No evaluation.")

    if args.total_steps <= 0:
        args.total_steps = args.num_epochs * len(train_loader)
    if args.warmup_steps <= 0:
        args.warmup_steps = int(0.1 * args.total_steps)  # warmup 10% steps
    if args.eval_steps <= 0:
        args.eval_steps = int(0.1 * args.total_steps)  # eval 10% steps

    # optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, 
        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
    
    # scheduler
    scheduler = get_scheduler(args.scheduler, optimizer, args.warmup_steps, args.total_steps)

    # amp
    scaler = GradScaler() if args.use_amp else None

    # tensorboard
    summary = SummaryWriter(args.tensorboard_path)

    # save args
    args_file = os.path.join(args.output_path, 'args.txt')
    logger.info(f"Freeze training args into: {args_file}")
    with open(args_file, 'w', encoding='utf8') as f:
        for name, value in vars(args).items():
            print(name, '=', value, sep=' ', file=f)
    
    for epoch in range(args.num_epochs):
        args.epoch = epoch
        logger.info(f"Start train {epoch} epoch...")

        start_time = time.time()
        train(model, train_loader, evaluator, optimizer, scheduler, scaler, summary, logger, args)
        during = time.time() - start_time
        logger.info(f"Training epoch {epoch} done during {during}s!")

        save_path = os.path.join(args.output_path, f"epoch_{epoch}")
        model.save_pretrained(save_path)

        if args.do_eval:
            mrr = evaluate(model, evaluator, summary, (epoch+1)*len(train_loader), logger, args,
                save_file=os.path.join(save_path, 'metrics.json'))

if __name__ == '__main__':
    main()
