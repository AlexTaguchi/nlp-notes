# Learning Protein Sequence Embeddings with NLP
# =============================================
# 
# Purpose: Train sequence embeddings to predict masked amino acids.
#
# Models:
# - Bidirectional LSTM
# - Transformer encoder
# - BERT

# Modules
from Bio import SeqIO
import gzip
import math
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.optim as optim

# Parameters
batch_size = 64
mask_ratio = 0.15
max_length = 512
# model = 'BiLSTM'
# model = 'TransformerEncoder'
model = 'BERT'

# Model settings
settings = {
    'BiLSTM': {
        'vocabulary': 25,
        'embedding_size': 100,
        'lstm_size': 64,
        'lstm_layers': 3,
        'dropout': 0.1
    },
    'TransformerEncoder': {
        'vocabulary': 25,
        'embedding_size': 200,
        'transformer_heads': 2,
        'transformer_size': 200,
        'transformer_layers': 2,
        'dropout': 0.1
    },
    'BERT': {
        'vocabulary': 25,
        'embedding_size': 200,
        'transformer_heads': 2,
        'transformer_size': 200,
        'transformer_layers': 2,
        'dropout': 0.1
    }
}


class BiLSTM(nn.Module):
    """Bidirectional LSTM encoder"""
    def __init__(self, vocabulary, embedding_size, lstm_size, lstm_layers, dropout=0.0):
        """Initialize bidirectional LSTM
        
        Arguments:
            vocabulary {int} -- Size of the amino acid vocabulary
            embedding_size {int} -- Size of amino acid embedding
            lstm_size {int} -- Size of LSTM encoder feedforward network
            lstm_layers {int} -- Number of LSTM encoder layers
        
        Keyword Arguments:
            dropout {float} -- Dropout rate (default: {0.0})
        """
        super(BiLSTM, self).__init__()

        # Amino acid embedder
        self.amino_embedder = nn.Embedding(vocabulary, embedding_size, padding_idx=0)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(embedding_size, lstm_size, lstm_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.encoder = nn.Linear(2 * lstm_size, embedding_size)

        # Linear decoder
        self.decoder = nn.Linear(embedding_size, vocabulary)

    def forward(self, sequences):
        # Sort sequences by length
        lengths = (sequences != 0).sum(dim=1)
        sequence_sort = torch.argsort(lengths, descending=True)
        sequence_unsort = torch.argsort(sequence_sort)
        sequences, lengths = sequences[sequence_sort], lengths[sequence_sort]

        # Embed sequences
        embedding = self.amino_embedder(sequences)
        embedding = pack_padded_sequence(embedding, lengths, batch_first=True)

        # Encode with bidirectional LSTM
        self.lstm.flatten_parameters()
        encoding, _ = self.lstm(embedding)
        encoding = pad_packed_sequence(encoding, batch_first=True, total_length=sequences.size(1))
        encoding = encoding[0][sequence_unsort]
        encoding = self.encoder(encoding)

        # Decode with linear layer
        decoding = self.decoder(encoding)

        return embedding, encoding, decoding


class TransformerEncoder(nn.Module):
    """Transformer encoder based on https://arxiv.org/pdf/1706.03762.pdf"""
    def __init__(self, vocabulary, embedding_size, transformer_heads,
                 transformer_size, transformer_layers, dropout=0.0):
        """Initialize transformer encoder
        
        Arguments:
            vocabulary {int} -- Size of the amino acid vocabulary
            embedding_size {int} -- Size of amino acid embedding
            transformer_heads {int} -- Number of transformer encoder attention heads
            transformer_size {int} -- Size of transformer encoder feedforward network
            transformer_layers {int} -- Number of transformer encoder layers
        
        Keyword Arguments:
            dropout {float} -- Dropout rate (default: {0.0})
        """
        super(TransformerEncoder, self).__init__()
        self.embedding_size = embedding_size

        # Amino acid embedder and positional encoder
        self.amino_embedder = nn.Embedding(vocabulary, embedding_size, padding_idx=0)
        self.positional_encoder = self.encode_positions()

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(embedding_size, transformer_heads,
                                                       transformer_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, transformer_layers)
        
        # Linear decoder
        self.decoder = nn.Linear(embedding_size, vocabulary)
    
    def encode_positions(self, max_length=5000):
        # Preallocate positional encoder
        positional_encoder = torch.zeros(max_length, self.embedding_size)

        # Calculate frequencies
        frequencies = torch.arange(0, self.embedding_size, 2).float()
        frequencies = torch.exp((-math.log(10000.0) / self.embedding_size) * frequencies)

        # Compute sines and cosines of positional encoder
        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        positional_encoder[:, 0::2] = torch.sin(positions * frequencies)
        positional_encoder[:, 1::2] = torch.cos(positions * frequencies)
        positional_encoder = positional_encoder.unsqueeze(0)

        return positional_encoder
    
    def forward(self, sequences):
        # Embed sequences with weighting factor
        embedding = self.amino_embedder(sequences) * math.sqrt(self.embedding_size)

        # Add positional encoding
        embedding += self.positional_encoder[:, :embedding.size(1), :]

        # Encode with transformer
        encoding = self.transformer_encoder(embedding.transpose(0, 1)).transpose(0, 1)

        # Decode with linear layer
        decoding = self.decoder(encoding)

        return embedding, encoding, decoding


class BERT(nn.Module):
    """BERT transformer based on https://arxiv.org/pdf/1810.04805.pdf"""
    def __init__(self, vocabulary, embedding_size, transformer_heads,
                 transformer_size, transformer_layers, dropout=0.1):
        """Initialize BERT
        
        Arguments:
            vocabulary {int} -- Size of the amino acid vocabulary
            embedding_size {int} -- Size of amino acid embedding
            transformer_heads {int} -- Number of transformer attention heads
            transformer_size {int} -- Size of transformer feedforward network
            transformer_layers {int} -- Number of transformer layers
        
        Keyword Arguments:
            dropout {float} -- Dropout rate (default: {0.0})
        """
        super(BERT, self).__init__()
        self.embedding_size = embedding_size

        # Amino acid embedder
        self.amino_embedder = nn.Embedding(vocabulary, embedding_size, padding_idx=0)
        
        # Transformer encoder
        encoder_norm = nn.LayerNorm(embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(embedding_size, transformer_heads,
                                                   transformer_size, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, transformer_layers, encoder_norm)

        # Transformer decoder
        decoder_norm = nn.LayerNorm(embedding_size)
        decoder_layer = nn.TransformerDecoderLayer(embedding_size, transformer_heads,
                                                   transformer_size, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, transformer_layers, decoder_norm)
    
    def forward(self, sequences):
        # Embed sequences
        embedding = self.amino_embedder(sequences)

        # Encode with transformer
        encoding = self.encoder(embedding.transpose(0, 1)).transpose(0, 1)

        # Decode with transformer
        decoding = self.decoder(embedding.transpose(0, 1), encoding.transpose(0, 1)).transpose(0, 1)

        return embedding, encoding, decoding


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for training and evaluating on protein sequence fasta files"""
    def __init__(self, file_path, mask_ratio=0, max_length=5000):
        self.file_path = file_path
        self.mask_ratio = mask_ratio
        self.max_length = max_length
        self.vocabulary = 25

        # Encode amino acids leaving 0 for padding token
        self.amino_encoder = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

        # Extract sequences from fasta
        self.sequences = [seq for seq in SeqIO.parse(gzip.open(self.file_path, 'rt'), format='fasta')
                          if len(seq) + 2 < self.max_length]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        # Encode sequence with unknown (21), start (22), and stop (23) tokens
        sequence = [22] + [self.amino_encoder.setdefault(aa, 21) for aa in self.sequences[i]] + [23]
        sequence = torch.tensor(sequence, dtype=torch.long)

        # Store correct sequence without start and stop tokens as label
        label = sequence.clone()

        # Randomly mask amino acids
        indices = random.sample(range(1, len(sequence)-1), int(self.mask_ratio * len(sequence)))
        mask = torch.zeros(len(sequence), dtype=torch.bool)
        for index in indices:

            # Mask amino acid
            mask[index] = True

            # Apply mask token (24) to amino acid 80% of the time, random mutation 10%, or nothing 10%
            if random.random() < 0.8:
                sequence[index] = 24
            elif random.random() < 0.5:
                sequence[index] = random.randint(1, 21)
                
        return sequence, mask, label


def collate_sequences(args):
    """Collate sequences, masks, and labels into padded tensors"""
    return [pad_sequence([arg[i] for arg in args], batch_first=True) for i in range(3)]


# Initialize dataset and data loader
dataset = SequenceDataset('.data/swiss-prot/uniprot_sprot.fasta.gz',
                          mask_ratio=mask_ratio, max_length=max_length)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_sequences)

# Select model
if model == 'BiLSTM':
    model = BiLSTM(**settings['BiLSTM'])
elif model == 'TransformerEncoder':
    model = TransformerEncoder(**settings['TransformerEncoder'])
elif model == 'BERT':
    model = BERT(**settings['BERT'])
else:
    raise ValueError('Model not found!')

# Set criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluate model
for i, batch in enumerate(data_loader):
    x, mask, y = batch
    _, _, output = model(x)
    optimizer.zero_grad()
    loss = criterion(output[mask], y[mask])
    loss.backward()
    optimizer.step()
    correct = torch.eq(y[mask], output[mask].argmax(dim=-1))
    accuracy = 100 * correct.sum().item() / len(correct)
    progress = 100 * ((i + 1) * batch_size) / len(dataset)
    print(f'Step {i + 1} ({progress:.3f}%): Accuracy = {accuracy:.1f}%')
