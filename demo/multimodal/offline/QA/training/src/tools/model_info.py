import sys
from transformers import AutoTokenizer, AutoModel, AutoConfig

model_dir = sys.argv[1]
config = AutoConfig.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#num_params = sum(p.numel() for p in model.parameters())
info = {
    'backbone': config.model_type,
    'num_layers': config.num_hidden_layers, 
    'num_heads': config.num_attention_heads,
    'hidden_dim': config.hidden_size,
    'ffnn_dim': config.intermediate_size,
    'max_seq_len': config.max_position_embeddings,
    'vocab_size': config.vocab_size,
    'num_params': '{}M'.format(int(num_params/1000000))
}
for k, v in info.items():
    print(k, v, sep='\t')

