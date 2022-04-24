import os
import logging
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from tqdm.autonotebook import trange
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator



def create_optimizer(model_params: List[Tensor], optimizer_class: Type[Optimizer] = transformers.AdamW, 
        optimizer_params : Dict[str, object]={}, weight_decay: float=0.01):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    return optimizer


def create_scheduler(optimizer, scheduler: str, warmup_steps: int, total_steps: int):
    """
    Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    """
    scheduler = scheduler.lower()
    if scheduler == 'constantlr':
        return transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    elif scheduler == 'warmupcosine':
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))


class Trainer(object):

    def __init__(self, model: SentenceTransformer, train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None, scheduler: str = 'WarmupLinear', warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW, optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01, max_grad_norm: float = 1,
            collate_fn: Callable=None, device=None, logger=None):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param model: SentenceTransformer model
        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param max_grad_norm: Used for gradient normalization.
        :param collate_fn: the collate function for dataloader.
        :param device: the device for training.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        #logger.info("Use pytorch device: {}".format(device))
        #logger.info("Num of objectives: {}".format(len(train_objectives)))
        #logger.info("Evaluator: {}".format(evaluator))

        self._model = model
        self._objectives = train_objectives
        self._evaluator = evaluator
        self._scheduler = scheduler
        self._warmup_steps = warmup_steps
        self._optim_class = optimizer_class
        self._optim_params = optimizer_params
        self._weight_decay = weight_decay
        self._max_grad_norm = max_grad_norm
        self._target_device = torch.device(device)
        self._collate_fn = collate_fn if collate_fn is not None else model.smart_batching_collate
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.best_score = -9999999

    def __str__(self):
        return str({
            'num_train_objectives': len(self._objectives),
            'scheduler': self._scheduler,
            'warmup_steps': self._warmup_steps,
            'optimizer': self._optim_class,
            'optimizer_params': self._optim_params,
            'device': self._target_device
        })

    def train(self,
            epochs: int = 1,
            steps_per_epoch = None,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0, 
            use_amp: bool = False,
            show_progress_bar: bool = True
            ):
        """
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param checkpoint_path: Folder to save checkpoints during training steps
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        self.best_score = -9999999
        dataloaders = []
        loss_models = []
        optimizers = []
        schedulers = []
        summary_writer = SummaryWriter(os.path.join(output_path, 'runs')) if output_path is not None else None

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self._model.to(self._target_device)

        loss_models = [loss for _, loss in self._objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in self._objectives]
        for dataloader in dataloaders:
            dataloader.collate_fn = self._collate_fn

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
        num_train_steps = int(steps_per_epoch * epochs)

        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            optimizer = create_optimizer(param_optimizer, self._optim_class, self._optim_params,
                weight_decay=self._weight_decay)
            scheduler = create_scheduler(optimizer, self._scheduler, warmup_steps=self._warmup_steps,
                total_steps=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler)

        skip_scheduler = False
        num_train_objectives = len(self._objectives)
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        self.logger.info(f"Num epochs: {epochs}")
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"Total steps: {num_train_steps}")
        self.logger.info(f"Num train objectives: {num_train_objectives}")
        self.logger.info(f"Steps of evaluation: {evaluation_steps}")
        self.logger.info(f"Steps of saving checkpoint: {checkpoint_save_steps}")
        self.logger.info(f"Save best model: {save_best_model}")
        self.logger.info("Save checkpoint: {}".format(checkpoint_path is not None))

        global_step = 0  # steps in all of epochs
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0  # steps in each epoch

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                loss_info = {}
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    # get inputs
                    features, labels = data

                    # forward and backward
                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), self._max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), self._max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                    loss_info[str(train_idx)] = loss_value.item()

                training_steps += 1
                global_step += 1

                # tensorboard
                if summary_writer is not None:
                    summary_writer.add_scalars("train-lossing", loss_info, global_step)

                # evaluation
                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    eval_score = self._eval_during_training(self._evaluator, output_path, save_best_model, epoch, training_steps)
                    if summary_writer is not None:
                        summary_writer.add_scalar("eval-metric", eval_score, global_step)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                # save checkpoint by step
                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            # end of each epoch
            self._eval_during_training(self._evaluator, output_path, save_best_model, epoch, -1)

        # save final step's checkpoint
        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

        # if best model cannot be saved, then we should save the final model
        if self._evaluator is None or not save_best_model:
            if output_path is not None:
                self._model.save(output_path)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback=None):
        """Runs evaluation during the training and save best model."""
        score = -1.0
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self._model, output_path=eval_path, epoch=epoch, steps=steps)
            #if callback is not None:
            #    callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model and output_path:
                    self._model.save(output_path)

        return score

    def _save_checkpoint(self, checkpoint_path, checkpoint_save_total_limit, step):
        """Store new and remove old checkpoint."""
        # Store new checkpoint
        self._model.save(os.path.join(checkpoint_path, str(step)))

        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                if subdir.isdigit():
                    old_checkpoints.append({'step': int(subdir), 'path': os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x['step'])
                shutil.rmtree(old_checkpoints[0]['path'])
