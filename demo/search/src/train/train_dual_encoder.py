import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

import torch
import transformers
#from transformers import AdamW
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from data import create_dual_encoder_dataloader
from modeling import TransformerDualEncoder
from losses import CosineSimilarityLoss, ContrastiveLoss, TripletLoss, ContrastiveInBatchLoss

#logger = logging.getLogger(__name__)

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

def get_loss(name, model, dual_model, **kwargs):
    if name == 'cosine':
        return CosineSimilarityLoss(model, dual_model, **kwargs)
    elif name == 'contrastive':
        return ContrastiveLoss(model, dual_model, **kwargs)
    elif name == 'triplet':
        return TripletLoss(model, dual_model, **kwargs)
    elif name == 'contrastive_in_batch':
        return ContrastiveInBatchLoss(model, dual_model, **kwargs)
    raise ValueError(f"Not support loss: {name}")

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
        default='pair_with_label',
        choices=['pair', 'pair_with_label', 'triplet'],
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
        '--model',
        type=str,
        default='DMetaSoul/sbert-chinese-general-v2',
        help='The pretrained model for training.'
    )
    parser.add_argument(
        '--dual-model',
        type=str,
        default='DMetaSoul/sbert-chinese-general-v2',
        help='The pretrained model for training.'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='cosine',
        choices=['cosine', 'contrastive', 'triplet', 'contrastive_in_batch']
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine_with_warmup',
        choices=['constant', 'constant_with_warmup', 'linear_with_warmup', 'cosine_with_warmup']
    )
    parser.add_argument(
        '--tied-model',
        action='store_true',
        help='The dual encoder will be tied weight.'
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

        features = [{k:v.cuda(args.gpu, non_blocking=True) for k,v in feat.items()} for feat in features]
        if labels is not None:
            labels = labels.cuda(args.gpu, non_blocking=True)

        if args.use_amp:
            with autocase():
                loss = model(features, labels)
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            skip_scheduler = scaler.get_scale() != scale_before_step
        else:
            loss = model(features, labels)
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
    q_model = model.model
    p_model = model.dual_model
    with torch.no_grad():
        res = evaluator.compute_metrices(q_model, p_model)
    precision = {f"precision@{k}":v for k,v in res['cos_sim']['precision@k'].items()}
    recall = {f"recall@{k}":v for k,v in res['cos_sim']['recall@k'].items()}
    mrr = {f"mrr@{k}":v for k,v in res['cos_sim']['mrr@k'].items()}
    logger.info(
        f"Eval Epoch: {args.epoch}\tSteps: {steps}\t"
        f"Precision: {precision}\tRecall: {recall}\tMRR: {mrr}"
    )
    if summary is not None:
        for name, value in precision.items():
            summary.add_scalar(name, value, steps)
        for name, value in recall.items():
            summary.add_scalar(name, value, steps)
        for name, value in mrr.items():
            summary.add_scalar(name, value, steps)
    if save_file is not None:
        with open(save_file, 'w', encoding='utf8') as f:
            f.write(json.dumps({
                "precision": precision,
                "recall": recall,
                "mrr": mrr
            }, ensure_ascii=False))
    return precision, recall, mrr


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

    # the encoder model
    model = TransformerDualEncoder.load_pretrained(args.model)

    if args.tied_model:
        dual_model = None
    else:
        dual_model = TransformerDualEncoder.load_pretrained(args.dual_model)

    # loss build on the top of encoder
    loss_model = get_loss(args.loss, model, dual_model)
    loss_model.cuda(args.gpu)
    #print(loss_model)

    # load train data
    label_index = int(args.train_label_index)
    label_converter = float if args.train_label_type == 'float' else int
    text_indices = [int(i) for i in args.train_text_index.split(',')]
    tokenizers = []
    tokenizers.append(model.tokenize)
    tokenizers.append(model.tokenize if dual_model is None else dual_model.tokenize)
    if len(text_indices) == 3:
        tokenizers.append(model.tokenize if dual_model is None else dual_model.tokenize)
    train_loader = create_dual_encoder_dataloader(args.train_file, args.train_kind, tokenizers, 
        text_indices=text_indices, label_index=label_index, label_converter=label_converter,
        shuffle=True, batch_size=args.train_batch_size)

    # check dataloader
    #for features, labels in train_loader:
    #    print(features, labels)
    #    break
    #exit()

    # load eval data
    evaluator = None
    if args.do_eval:
        eval_queries, eval_corpus, eval_relevant = {}, {}, {}
        for qid, query in load_tsv(args.eval_qid_file):
            eval_queries[qid] = query
        for pid, para in load_tsv(args.eval_pid_file):
            eval_corpus[pid] = para
        for qid, pids in load_tsv(args.eval_rel_file):
            pids = pids.split(',')
            eval_relevant[qid] = set(pids)
        top_k_list = [1, 10, 50]
        evaluator = InformationRetrievalEvaluator(eval_queries, eval_corpus, eval_relevant, 
            batch_size=args.eval_batch_size, corpus_chunk_size=args.eval_batch_size*10, 
            mrr_at_k=top_k_list, ndcg_at_k=top_k_list, accuracy_at_k=top_k_list, precision_recall_at_k=top_k_list, map_at_k=top_k_list,  
            show_progress_bar=True, write_csv=False)

    if args.total_steps <= 0:
        args.total_steps = args.num_epochs * len(train_loader)
    if args.warmup_steps <= 0:
        args.warmup_steps = int(0.1 * args.total_steps)  # warmup 10% steps
    if args.eval_steps <= 0:
        args.eval_steps = int(0.1 * args.total_steps)  # eval 10% steps

    # create optimizer
    param_optimizer = list(loss_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, 
        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
    
    # create scheduler
    scheduler = get_scheduler(args.scheduler, optimizer, args.warmup_steps, args.total_steps)

    # amp
    scaler = GradScaler() if args.use_amp else None

    # tensorboard
    summary = SummaryWriter(args.tensorboard_path)

    # save args
    with open(os.path.join(args.output_path, 'args.txt'), 'w', encoding='utf8') as f:
        for name, value in vars(args).items():
            print(name, '=', value, sep=' ', file=f)
    
    for epoch in range(args.num_epochs):
        args.epoch = epoch
        logger.info(f"Start train {epoch} epoch...")

        if args.do_eval:
            precision, recall, mrr = evaluate(loss_model, evaluator, summary, (epoch)*len(train_loader), logger, args)

        start_time = time.time()
        train(loss_model, train_loader, evaluator, optimizer, scheduler, scaler, summary, logger, args)
        during = time.time() - start_time
        logger.info(f"Training epoch {epoch} done during {during}s!")

        save_path = os.path.join(args.output_path, f"epoch_{epoch}")
        loss_model.save(save_path)

        if args.do_eval:
            precision, recall, mrr = evaluate(loss_model, evaluator, summary, (epoch+1)*len(train_loader), logger, args,
                save_file=os.path.join(save_path, 'metrics.json'))


if __name__ == '__main__':
    main()
