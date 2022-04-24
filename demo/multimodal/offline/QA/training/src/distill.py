"""
adopted from: https://www.sbert.net/examples/training/distillation/README.html

Knowledge Distillation: We use a well working teacher model to train a fast and light student model. The student model learns to imitate the produced sentence embeddings from the teacher. We train this on a diverse set of sentences we got from SNLI + Multi+NLI + Wikipedia.

After the distillation is finished, the student model produce nearly the same embeddings as the teacher, however, it will be much faster.

The script implements to options two options to initialize the student:
Option 1: Train a light transformer model like TinyBERT to imitate the teacher
Option 2: We take the teacher model and keep only certain layers, for example, only 4 layers.

Option 2) works usually better, as we keep most of the weights from the teacher. In Option 1, we have to tune all weights in the student from scratch.

There is a performance - speed trade-off. However, we found that a student with 4 instead of 12 layers keeps about 99.4% of the teacher performance, while being 2.3 times faster.
"""
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, evaluation
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.datasets import ParallelSentencesDataset
import logging
from datetime import datetime
import os
import gzip
import csv
import random
from sklearn.decomposition import PCA
import torch
import argparse

from dataset import CorpusDataset, STSDataset

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
logger = logging.getLogger(__name__)


def load_teacher_model(model_name_or_path):
    """Teacher Model: Model we want to distill to a smaller model"""
    teacher_model = SentenceTransformer(model_name_or_path)
    return teacher_model


def load_student_model(model_name_or_path, teacher_model_name_or_path=None, layers_to_keep=[]):
    """Student Model: there are two options to create a light and fast student model"""
    if teacher_model_name_or_path is not None and len(layers_to_keep) > 0:
        # 1) Create a smaller student model by using only some of the teacher layers
        student_model = SentenceTransformer(teacher_model_name_or_path)

        # Get the transformer model
        auto_model = student_model._first_module().auto_model

        # Which layers to keep from the teacher model. We equally spread the layers to keep over the original teacher
        logging.info("Remove layers from student. Only keep these layers: {}".format(layers_to_keep))
        new_layers = torch.nn.ModuleList([layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep])
        auto_model.encoder.layer = new_layers
        auto_model.config.num_hidden_layers = len(new_layers)
    else:
        # 2) The other option is to train a small model like TinyBERT to imitate the teacher.
        word_embedding_model = models.Transformer(model_name_or_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return student_model


def teacher_match_student(teacher_model, student_model, pca_sentences):
    t_dim = teacher_model.get_sentence_embedding_dimension()
    s_dim = student_model.get_sentence_embedding_dimension()
    assert s_dim <= t_dim, "Student's dim must be less or equal to teacher!"
    if s_dim < t_dim:
        # Student model has fewer dimensions. Compute PCA for the teacher to reduce the dimensions
        pca_embeddings = teacher_model.encode(pca_sentences, convert_to_numpy=True)
        pca = PCA(n_components=s_dim)
        pca.fit(pca_embeddings)

        #Add Dense layer to teacher that projects the embeddings down to the student embedding size
        dense = models.Dense(in_features=t_dim, out_features=s_dim, bias=False, activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
        teacher_model.add_module('dense', dense)
    else:
        return teacher_model


parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", required=True, help="Experiment Name.")
parser.add_argument("--train-file", required=True, help="Train corpus file.")
parser.add_argument("--dev-file", required=True, help="Dev corpus file.")
parser.add_argument("--test-file", default="", help="Eval Dataset file, it is a like-sts data.")
parser.add_argument("--teacher-model", required=True, help="The pre-trained teacher sbert model name or checkpoint path for teaching the student model.")
#parser.add_argument("--teacher-local", default=1, type=int, choices=[0,1], help="If the teacher model is on local machine.")
parser.add_argument("--student-model", default="", help="The pre-trained student model name or checkpoint path to init the student model.")
#parser.add_argument("--student-local", default=1, type=int, choices=[0,1], help="If the student model is on local machine.")
parser.add_argument("--student-keep-layers", default="", help="The student model would keep layers of teacher to init itself")
parser.add_argument("--max-seq-len", type=int, default=256, help="The max length of sequence.")
parser.add_argument("--device", default="cuda:0", help="The cuda device for training")
parser.add_argument("--num-epochs", type=int, default=4, help="The num of epochs.")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="The initial learning rate.")
parser.add_argument("--train-batch-size", type=int, default=256, help="The batch size for training")
parser.add_argument("--eval-batch-size", type=int, default=16, help="The batch size for evaluation")
parser.add_argument("--warmup-rate", type=float, default=0.1, help="The rate of warmup steps.")
parser.add_argument("--eval-rate", type=float, default=0.1, help="The rate of eval steps.")
parser.add_argument("--output-dir", default="./output", help="The output directory.")
parser.add_argument("--model-save-dir", default="", help="The model saved directory.")
args = parser.parse_args()


exp_name = args.exp_name
num_epochs = args.num_epochs
max_seq_len = args.max_seq_len
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
inference_batch_size = train_batch_size
warmup_rate = args.warmup_rate
eval_rate = args.eval_rate
learning_rate = args.learning_rate
if args.model_save_dir:
    output_path = args.model_save_dir
else:
    output_path = os.path.join(args.output_dir, 
        'training_{}'.format(exp_name),
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
logger.info(f"Output dir: {output_path}")

# We train the student_model such that it creates sentence embeddings similar to the embeddings from the teacher_model
# For this, we need a large set of sentences. These sentences are embedded using the teacher model,
# and the student tries to mimic these embeddings. It is the same approach as used in: https://arxiv.org/abs/2004.09813
train_sents = CorpusDataset(args.train_file).load(shuffle=True)
logger.info("Train Corpus size: {}".format(len(train_sents)))

# init teacher and student model
teacher_model = load_teacher_model(args.teacher_model)
student_model = load_student_model(args.student_model
    , args.teacher_model, [int(i) for i in args.student_keep_layers.split(',')])
teacher_model = teacher_match_student(teacher_model, student_model, train_sents[:40000])
logger.info("Teacher: {}".format(teacher_model))
logger.info("Student: {}".format(student_model))

# Create train data loader and loss
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model
    , batch_size=inference_batch_size, use_embedding_cache=False)
train_data.add_dataset([[sent] for sent in train_sents], max_sentence_length=max_seq_len)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)

# We create an evaluator, that measure the Mean Squared Error (MSE) between the teacher and the student embeddings
dev_sents = CorpusDataset(args.dev_file).load()
dev_evaluator_mse = evaluation.MSEEvaluator(dev_sents, dev_sents, teacher_model=teacher_model, batch_size=eval_batch_size, name='{}-eval-mse'.format(exp_name))
logger.info("Dev Corpus size: {}".format(len(dev_sents)))

dev_evaluator_sts = None
if args.test_file:
    # We use the STS benchmark dataset to measure the performance of student model im comparison to the teacher model
    dev_samples = STSDataset(args.test_file).load()
    dev_evaluator_sts = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, 
        batch_size=eval_batch_size, name='{}-eval-sts'.format(exp_name))
    logger.info("Test size: {}".format(len(dev_samples)))
    logging.info("Teacher Performance(before): {}".format(dev_evaluator_sts(teacher_model)))
    logging.info("Student Performance(before): {}".format(dev_evaluator_sts(student_model)))

if dev_evaluator_sts is not None:
    evaluator = evaluation.SequentialEvaluator([dev_evaluator_sts, dev_evaluator_mse])
else:
    evaluator = dev_evaluator_mse

# Train the student model to imitate the teacher
warmup_steps = len(train_dataloader) * num_epochs * warmup_rate
eval_steps = len(train_dataloader) * num_epochs * eval_rate
student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  evaluation_steps=eval_steps,
                  output_path=output_path,
                  save_best_model=True,
                  optimizer_params={'lr': learning_rate, 'eps': 1e-6, 'correct_bias': False},
                  use_amp=True)

if dev_evaluator_sts is not None:
    logging.info("Teacher Performance(after): {}".format(dev_evaluator_sts(teacher_model)))
    logging.info("Student Performance(after): {}".format(dev_evaluator_sts(student_model)))
