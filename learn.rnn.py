import torch
from torch import nn

inputSize = 1
hiddenSize = 20
numLayers = 2
nonLinearity = 'tanh'
bias = True
batch1st = True
dropout = 0.0
bidirectional = False
bidirectional=False
device=None
dtype=None

batchSize = 32
sequenceLength = 5

rnn = nn.RNN(inputSize, hiddenSize, numLayers, batch_first=batch1st)
h0 = torch.randn( numLayers, batchSize, hiddenSize )

dinput = torch.randn(batchSize, sequenceLength, inputSize)

doutput, ht = rnn(dinput)



'''
input: tensor of shape
( L , Hin)
(L,H in) for unbatched input,
( L , N , Hin)
(L,N,Hin) when batch_first=False or
( N , L , Hin) when batch_first=True 

containing the features of the input sequence. 
The input can also be a packed variable length sequence. 
See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.

hx: tensor of shape
( D ∗ num_layers , Hout) (D∗num_layers,H out) for unbatched input or
( D ∗ num_layers , N , Hout) (D∗num_layers,N, Hout) 
containing the initial hidden state for the input sequence batch. Defaults to zeros if not provided.

where:

N = batch size
L = sequence length
D = 2
 if bidirectional=True 
 otherwise  = 1

Hin = input_size
Hout = hidden_size

N= batch size
L= sequence length
D= 2 if bidirectional=True otherwise 1
Hin = input_size
Hout = hidden_size



'''

