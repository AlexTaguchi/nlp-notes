# Modules
from Bio import SeqIO
from gzip import open as gzopen
import numpy as np
import torch

class ProteinDataset(torch.utils.data.Dataset):
    """Dataset for training and evaluating on protein fasta files"""
    def __init__(self, file_path, mask_ratio=0):
        self.file_path = file_path
        self.mask_ratio = mask_ratio

        # Create amino acid encoder
        self.amino_acids = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
        numeric = np.arange(len(self.amino_acids))
        numeric[21:] = [11, 4, 20, 20]
        numeric += 1
        self.encoder = {aa: n for aa, n in zip(self.amino_acids, numeric)}

        # Extract sequences from file
        self.sequences = [seq for seq in SeqIO.parse(gzopen(self.file_path, 'rt'), format='fasta')]
        # for :
        #     self.sequences.append(record.seq)
            # print([aa for aa in record])
            # self.sequences.append()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        return torch.tensor([self.encoder[aa] if aa in self.amino_acids
                             else 0 for aa in self.sequences], dtype=torch.long)
        # pass
        # return (*mask_tokens(self.sequences[i], self.mask_ratio), self.labels[i])

dataset = ProteinDataset('.data/swiss-prot/uniprot_sprot.fasta.gz')
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=512, collate_fn=collate_sequences)


print(len(a))
# Read Swiss-Prot sequences
# sequences = SeqIO.parse(gzopen('.data/swiss-prot/uniprot_sprot.fasta.gz', 'rt'), format='fasta')
# for sequence in sequences:
#     print(sequence.seq)