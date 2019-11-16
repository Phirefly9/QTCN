import torch.nn.functional as F
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(LSTM, self).__init__()
        hidden_size = 512
        bidirectional = False
        linear_size = hidden_size * (2 if bidirectional else 1)
        print(linear_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(linear_size, output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        
        y1 = self.lstm(inputs)  # input dimension (N, C_in, L_in), output (n, num_chan[-1], L_in) 
        #print(tmp.shape)
        o = self.linear(y1[0][:, -1, :]) # input dimension (N, num_chan[-1], L_in), output (n, C_out[-1]) 
        
        return F.log_softmax(o, dim=1)