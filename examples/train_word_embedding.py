# Next Token Prediction with Transformers
# =======================================
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# 
# Purpose: Train the transformer model described in "Attention Is All You Need"
# (https://arxiv.org/pdf/1706.03762.pdf) on a language modeling task to predict
# the next token in a sequence.
#
# Model:
# - Embeddings are generated for each token in the a sequence
# - Positional encodings are added to the token embeddings
# - Square attention mask only allows tokens to see previous positions
# - Embeddings and attention mask are passed through Transformer encoder layers
# - Linear layer predicts the next token

# Modules
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchtext
from torchtext.data.utils import get_tokenizer

# Parameters
batch_train = 20
batch_evaluate = 10
chunks = 35
dropout = 0.2
epochs = 3
token_embedding = 200
transformer_heads = 2
transformer_dimensions = 200
transformer_layers = 2


class TransformerModel(nn.Module):
    """Transformer model based on https://arxiv.org/pdf/1706.03762.pdf"""
    def __init__(self, vocabulary, token_embedding, transformer_heads,
                 transformer_dimensions, transformer_layers, dropout=0.5):
        """Initialize transformer model
        
        Arguments:
            vocabulary {int} -- Size of the token vocabulary
            token_embedding {int} -- Size of token embedding
            transformer_heads {int} -- Number of transformer encoder attention heads
            transformer_dimensions {int} -- Dimension of transformer encoder feedforward network
            transformer_layers {int} -- Number of transformer encoder layers
        
        Keyword Arguments:
            dropout {float} -- Dropout rate (default: {0.5})
        """
        super(TransformerModel, self).__init__()
        self.token_embedding = token_embedding

        # Embedding and positional encoding of tokens
        self.embedder = nn.Embedding(vocabulary, token_embedding)
        self.positional_encoder = PositionalEncoding(token_embedding, dropout)

        # Transformer encoder layers
        transformer_layer = TransformerEncoderLayer(token_embedding, transformer_heads,
                                                    transformer_dimensions, dropout)
        self.transformer_encoder = TransformerEncoder(transformer_layer, transformer_layers)
        
        # Linear layer for predicting next token
        self.linear = nn.Linear(token_embedding, vocabulary)

        # Initialize weights
        self.init_weights()

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = 0.1
        self.embedder.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def forward(self, sequences):
        self.src_mask = self._generate_square_subsequent_mask(len(sequences)).to(sequences.device)
        print(sequences.size())
        sequences = self.embedder(sequences) * math.sqrt(self.token_embedding)
        print(sequences.size())
        sequences = self.positional_encoder(sequences)
        print(sequences.size())
        output = self.transformer_encoder(sequences, self.src_mask)
        print(output.size())
        output = self.linear(output)
        print(output.size())
        e
        return output


class PositionalEncoding(nn.Module):
    """Add sinusoidal positional information about the tokens"""
    def __init__(self, token_embedding, dropout=0.1, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encoding = torch.zeros(max_length, token_embedding)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, token_embedding, 2).float() *
                                  (-math.log(10000.0) / token_embedding))
        positional_encoding[:, 0::2] = torch.sin(position * division_term)
        positional_encoding[:, 1::2] = torch.cos(position * division_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(0), :]
        return self.dropout(x)


def batch_text(text, batches):
    """Truncate text to a multiple of batches and reshape into batched tensor
    
    Arguments:
        text {generator} -- Text generator
        batches {int} -- Token batch size
    
    Returns:
        tensor -- Text reshaped into rectangular tensor
    """
    text = wiki_text.numericalize([text.examples[0].text])
    text = text.narrow(0, 0, (text.size(0) // batches) * batches)
    return text.view(batches, -1).t().contiguous()

def get_batch(text, batch):
    """Generate input and target sequence for batch
    
    Arguments:
        text {tensor} -- Text tensor
        batch {[type]} -- Batch number to call
    
    Returns:
        tuple -- Data and target tensors
    """
    length = min(chunks, len(text) - 1 - batch)
    data = text[batch:batch+length]
    target = text[batch+1:batch+1+length].view(-1)
    return data, target

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import Wikitext-2 dataset
wiki_text = torchtext.data.Field(tokenize=get_tokenizer('basic_english'),
                                 init_token='<sos>', eos_token='<eos>', lower=True)
train_txt, validation_txt, test_txt = torchtext.datasets.WikiText2.splits(wiki_text)
wiki_text.build_vocab(train_txt)

# Batch Wikitext-2 for training and evaluation
train_data = batch_text(train_txt, batch_train).to(device)
validation_data = batch_text(validation_txt, batch_evaluate).to(device)
test_data = batch_text(test_txt, batch_evaluate).to(device)

# Instantiate model
vocabulary = len(wiki_text.vocab.stoi)
model = TransformerModel(vocabulary, token_embedding, transformer_heads,
                         transformer_dimensions, transformer_layers, dropout).to(device)

# Set criterion and optimizer for learning
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5.0)

# Adjust learning rate through epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train():
    """Train model"""
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, chunks)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, vocabulary), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed_time = time.time() - start_time
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_data) // chunks:5d} batches | lr '
                  f'{scheduler.get_lr()[0]:02.2f} | {elapsed_time * 1000 / log_interval:5.2f}'
                  f' ms/batch | loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model, validation_data):
    """Evaluate model"""
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, validation_data.size(0) - 1, chunks):
            data, targets = get_batch(validation_data, i)
            output = model(data)
            output_flat = output.view(-1, vocabulary)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(validation_data) - 1)

# Train over multiple epochs
best_model = None
best_validation_loss = float('inf')
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, validation_data)
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}')
    print('-' * 89)

    # Save model if best validation loss observed thus far
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        best_model = model

    # Adjust learning rate
    scheduler.step()

# Evaluate the model on the test dataset
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
print('=' * 89)