import os
import sys
import logging
import argparse
from datetime import datetime

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.readers import InputExample
from transformers.optimization import AdamW

from config import TaskConfig
from trainer import Trainer
from modeling import create_encoder, create_evaluator


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", required=True, help="Experiment Name.")
parser.add_argument("--task-type", required=True, 
    choices=TaskConfig.task_list(), help="The type of task determine the dataset and loss type.")
parser.add_argument("--loss-type", default="default", help="The loss type of this task(multi-losses split by comma).")
parser.add_argument("--params-file", help="The training and losses custom hyper-params json file, if you don't provide this all params will be default.")
parser.add_argument("--train-file", required=True, help="Train Dataset file.")
parser.add_argument("--dev-file", required=True, help="Eval/dev Dataset file, it is a like-sts data.")
parser.add_argument("--test-file", default="", help="Test Dataset file, it is a like-sts data.")
parser.add_argument("--dev-type", default="sts", choices=["sts", "qmc", "nli"], help="The dev/eval data type.")
parser.add_argument("--test-type", default="sts", choices=["sts", "qmc", "nli"], help="The test data type.")
parser.add_argument("--model", default="bert-base-uncased", help="The BERT model name or path.")
parser.add_argument("--is-pretrained-model", type=int, default=0, choices=[0,1], help="If the model is a sbert pretrained.")
parser.add_argument("--max-seq-len", type=int, default=256, help="The max length of sequence.")
parser.add_argument("--device", default="cuda:0", help="The cuda device for training.")
parser.add_argument("--pooling", default="mean", help="The pooling method of encoder.") 
parser.add_argument("--num-epochs", type=int, default=4, help="The num of epochs.")
parser.add_argument("--warmup-rate", type=float, default=0.1, help="The factor of warmup steps in all epochs.")
parser.add_argument("--eval-rate", type=float, default=0.1, help="The factor of evaluate steps in each epoch.")
parser.add_argument("--learning-rate", type=float, default=2e-05, help="The initial learning rate.")
parser.add_argument("--train-batch-size", type=int, default=256, help="The batch size for training")
parser.add_argument("--eval-batch-size", type=int, default=128, help="The batch size for evaluation")
parser.add_argument("--output-dir", default="./output", help="The output directory.")
parser.add_argument("--model-save-dir", default="", help="The model saved directory.")
args = parser.parse_args()

# arguments
exp_name = "{}-{}_{}".format(args.exp_name, args.task_type, args.loss_type)
model_name_or_path = args.model
num_epochs = args.num_epochs
is_pretrained_model = args.is_pretrained_model == 1
if args.model_save_dir:
    model_save_path = args.model_save_dir
else:
    model_save_path = os.path.join(args.output_dir, 
        'training_{}'.format(exp_name),
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
logger.info(f"Experiment: {exp_name}")
logger.info(f"Save Path: {model_save_path}")

# Create task config
task_cfg = TaskConfig(args.task_type, args.params_file)
if args.loss_type != "default":
    task_cfg.clear_losses()
    task_cfg.add_losses_from_str(args.loss_type, ',')
task_losses = task_cfg.get_losses()

# Create encoder model
model = create_encoder(model_name_or_path=model_name_or_path, 
    max_seq_len=args.max_seq_len, device=args.device, 
    is_pretrained_model=is_pretrained_model, pooling=args.pooling)
logger.info("Model: {}".format(model))
logger.info("Device: {}".format(model._target_device))

# Load train data
train_dataloaders = []
train_datasets = task_cfg.load_dataset(args.train_file, 
    losses=task_losses)
for i, train_samples in enumerate(train_datasets):
    logger.info("Train dataset-{} size: {}".format(i, len(train_samples)))
    train_dataloaders.append(DataLoader(train_samples, shuffle=True, 
        batch_size=args.train_batch_size))

# Create multi-task losses
train_losses = []
for loss_name in task_losses:
    loss_class = task_losses[loss_name]['class']
    loss_kwargs = task_losses[loss_name]['kwargs']
    loss = loss_class(model=model, **loss_kwargs)
    logger.info("Train loss-{}: {}".format(len(train_losses), loss))
    train_losses.append(loss)

assert len(train_dataloaders) == len(train_losses), "Multi-task learning must with same number datasets and losses."

# Load dev data and evaluator
if args.dev_file:
    # We add an evaluator, which evaluates the performance during training
    dev_samples, evaluator = create_evaluator('{}-dev'.format(exp_name), args.dev_file, 
        task_type=args.dev_type, batch_size=args.eval_batch_size, model=train_losses[0]) # model is encoder+loss
else:
    dev_samples, evaluator = [], None
logger.info("Dev dataset size: {}".format(len(dev_samples)))
logger.info("Dev evaluator: {}".format(evaluator))

# Train the model
total_steps = min([len(dl) for dl in train_dataloaders]) * num_epochs
eval_steps = int(total_steps * args.eval_rate)  # evalulate in each eval_steps
warmup_steps = int(total_steps * args.warmup_rate) #10% of train data for warm-up
# We can pass more than one tuple in order to perform multi-task learning on several datasets with different loss functions.
# 1. training by sentence-transformers
#model.fit(train_objectives=list(zip(train_dataloaders, train_losses)),
#          evaluator=evaluator,
#          epochs=num_epochs,
#          warmup_steps=warmup_steps,
#          evaluation_steps=eval_steps,
#          #optimizer_class=AdamW,
#          #optimizer_params={'correct_bias': False, 'eps': 1e-06, 'lr': args.learning_rate},
#          optimizer_params = {'lr': args.learning_rate},
#          output_path=model_save_path)
# 2. training by trainer
trainer = Trainer(model,
    train_objectives=list(zip(train_dataloaders, train_losses)),
    evaluator=evaluator,
    #scheduler='WarmupLinear',
    warmup_steps=warmup_steps,
    optimizer_params = {'lr': args.learning_rate},
    device=args.device,
    logger=logger
)
trainer.train(epochs=num_epochs, evaluation_steps=eval_steps, output_path=model_save_path)

# Evaluate model
if args.test_file:
    test_samples, test_eval = create_evaluator('{}-test'.format(exp_name), args.test_file, 
        task_type=args.test_type, batch_size=args.eval_batch_size, model=train_losses[0])
    logger.info("Test dataset size: {}".format(len(test_samples)))
    logger.info("Test evaluator: {}".format(test_eval))
    model = SentenceTransformer(model_save_path) # load model from disk
    test_eval(model, output_path=model_save_path)
