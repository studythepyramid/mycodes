import torch

nn = torch.nn

# (input size, hidden size, number of layers)
rnn = nn.LSTM(10, 20, 2)

input = torch.randn(5, 8, 10)
h0 = torch.randn(2, 8, 20)
c0 = torch.randn(2, 8, 20)

output, (hn, cn) = rnn(input, (h0, c0))
