import torch.nn.functional as F
from torch import nn
from QTCN.qtcn.qtcn import QTemporalConvNet

class QTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(QTCN, self).__init__()
        self.tcn = QTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        
        y1 = self.tcn(inputs)  # input dimension (N, C_in, L_in), output (n, num_chan[-1], L_in)
        o = self.linear(y1[:, :, -1]) # input dimension (N, num_chan[-1], L_in), output (n, C_out[-1]) 
        return F.log_softmax(o, dim=1)