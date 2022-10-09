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
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import create_text_classification_dataloader
from modeling import TextClassificationModel 


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='The name of experiment'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-cased',
        help='The pretrained model for training.'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help="The train data file."
    )
    parser.add_argument(
        '--eval-file',
        type=str,
        default='',
        help='The dev data file.'
    )
    parser.add_argument(
        '--num-labels',
        type=int,
        default=15,
        help='The number of class labels.'
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


def train(model, train_loader, eval_loader, optimizer, scheduler, scaler, summary, logger, args):
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
            evaluate(model, eval_loader, summary, train_steps, logger, args)

        if args.save_steps > 0 and train_steps % args.save_steps == 0:
            save_path = os.path.join(args.output_path, f"step_{train_steps}")
            model.save(save_path)


def evaluate(model, eval_loader, summary, steps, logger, args, save_file=None):
    """Should be implemented"""
    if not args.do_eval or eval_loader is None:
        return
    model.eval()
    total_loss = 0.0
    total_sample = 0
    with torch.no_grad():
        for _, (features, labels) in enumerate(tqdm(eval_loader)):
            features = {k:v.cuda(args.gpu, non_blocking=True) for k,v in features.items()}
            if labels is not None:
                labels = labels.cuda(args.gpu, non_blocking=True)
            _, loss = model(**features, labels=labels)
            total_loss += loss.item() * len(labels)
            total_sample += len(labels)
            
    eval_loss = total_loss / total_sample
    logger.info(f"Eval loss: {eval_loss:.6f}")
    if summary is not None:
        summary.add_scalar("eval_loss", eval_loss, steps)
                
    model.train()
    return


def main():
    args = parse_args()

    # output
    args.output_path = os.path.join(args.output_path, 
        args.name,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(args.output_path, exist_ok=True)

    # logger
    if not args.log_file:
        args.log_file = os.path.join(args.output_path, 'logs.txt')
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(args.log_file, args.log_level)

    if not args.tensorboard_path:
        args.tensorboard_path = os.path.join(args.output_path, 'runs')

    if os.path.isfile(args.eval_file):
        args.do_eval = True
    else:
        args.do_eval = False

    # model
    model = TextClassificationModel(args.model, 
        num_labels=args.num_labels, task_type='binary_classification')
    model.cuda(args.gpu)

    # train&eval data
    train_loader = create_text_classification_dataloader(args.train_file, model.preprocess, 
        text_indices=[0], label_index=2, label_converter=int, 
        batch_size=args.train_batch_size, shuffle=True)
    logger.info(f"Train data batches: {len(train_loader)}")

    eval_loader = None
    if args.do_eval:
        eval_loader = create_text_classification_dataloader(args.eval_file, model.preprocess, 
            text_indices=[0], label_index=2, label_converter=int, 
            batch_size=args.train_batch_size, shuffle=True)
        logger.info(f"Eval data batches: {len(eval_loader)}")
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
        train(model, train_loader, eval_loader, optimizer, scheduler, scaler, summary, logger, args)
        during = time.time() - start_time
        logger.info(f"Training epoch {epoch} done during {during}s!")

        save_path = os.path.join(args.output_path, f"epoch_{epoch}")
        model.save(save_path)

if __name__ == '__main__':
    main()
