import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TimeSeriesDataset(Dataset):

    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    # sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)