"""
copy from https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/MLM/README.md

Note: Only running MLM will not yield good sentence embeddings. But you can first tune your favorite transformer model with MLM on your domain specific data. Then you can fine-tune the model with the labeled data you have or using other data sets like NLI, Paraphrases, or STS.
"""
import sys
import gzip
import argparse
from datetime import datetime
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments

from sentence_transformers import SentenceTransformer
from modeling import create_evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", required=True, help="Experiment Name.")
parser.add_argument("--task-type", default='single')
parser.add_argument("--loss-type", default='mlm')
parser.add_argument("--model", required=True, help="The model name or path.")
parser.add_argument("--train-file", required=True, help="Train Corpus file.")
parser.add_argument("--dev-file", default='', help="Dev Corpus file (optional).")
parser.add_argument("--test-file", default='', help="The downstream task to be evaluated.")
parser.add_argument("--test-type", default="sts", choices=["sts", "qmc", "nli"], help="The eval data type.")
parser.add_argument("--num-epochs", type=int, default=4, help="The num of epochs.")
parser.add_argument("--train-batch-size", type=int, default=256, help="The batch size for training")
parser.add_argument("--eval-batch-size", type=int, default=256, help="The batch size for evaluation")
parser.add_argument("--max-seq-len", type=int, default=256, help="The max length of sequence.")
parser.add_argument("--learning-rate", type=float, default=2e-05, help="The initial learning rate.")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--do-whole-word-mask", action='store_true', help="enable whole-word-mask for MLM.")
parser.add_argument("--mlm-prob", type=float, default=0.15, help="the mask probability of MLM.")
parser.add_argument("--output-dir", default="./output", help="The output directory.")
parser.add_argument("--model-save-dir", default="", help="The model saved directory.")
args = parser.parse_args()


exp_name = "{}-{}_{}".format(args.exp_name, args.task_type, args.loss_type)
model_name = args.model
per_device_train_batch_size = args.train_batch_size
num_train_epochs = args.num_epochs            #Number of epochs
max_length = args.max_seq_len                #Max length for a text input
do_whole_word_mask = args.do_whole_word_mask       #If set to true, whole words are masked
mlm_prob = args.mlm_prob                #Probability that a word is replaced by a [MASK] token
learning_rate = args.learning_rate

save_steps = 1000               #Save model every 1k steps
use_fp16 = False                #Set to True, if your GPU supports FP16 operations
if args.model_save_dir:
    output_dir = args.model_save_dir
else:
    output_dir = os.path.join(args.output_dir, 'training_{}'.format(exp_name), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Save checkpoints to:", output_dir)


# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load datasets
train_path = args.train_file
train_sentences = []
with gzip.open(train_path, 'rt', encoding='utf8') if train_path.endswith('.gz') else  open(train_path, 'r', encoding='utf8') as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)
print("Train sentences:", len(train_sentences))

dev_sentences = []
if args.dev_file:
    dev_path = args.dev_file
    with gzip.open(dev_path, 'rt', encoding='utf8') if dev_path.endswith('.gz') else open(dev_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            line = line.strip()
            if len(line) >= 10:
                dev_sentences.append(line)
print("Dev sentences:", len(dev_sentences))

#A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None

# Training
if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps" if dev_dataset is not None else "no",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=save_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=use_fp16, 
    learning_rate=learning_rate
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

print("Training...")
trainer.train()
print("Training done")

print("Save model to:", output_dir)
model.save_pretrained(output_dir)

if args.test_file:
    test_samples, test_eval = create_evaluator('{}-test'.format(exp_name), args.test_file, 
        task_type=args.test_type, batch_size=args.eval_batch_size)
    print("Test size: {}".format(len(test_samples)))
    model = SentenceTransformer(output_dir) # load model from disk
    test_eval(model, output_path=output_dir)
