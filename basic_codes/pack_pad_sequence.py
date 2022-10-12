import torch
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

data = [torch.tensor([9]),
        torch.tensor([1,2,3,4]),
        torch.tensor([5,6])]


seq_len = [s.size(0) for s in data]
data = pad_sequence(data, batch_first=True)
data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=False)
print(data)
