import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from QTCN.qtcn.quaternion_layers import QuaternionConv
from QTCN.tcn.tcn import TemporalBlock
from torch.nn.utils.weight_norm import WeightNorm

class QChomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(QChomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class QTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(QTemporalBlock, self).__init__()
        
        self.conv1 = QuaternionConv(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilatation=dilation, operation='convolution1d', bias=True)
        
        WeightNorm.apply(self.conv1, "r_weight", 0)
        WeightNorm.apply(self.conv1, "i_weight", 0)
        WeightNorm.apply(self.conv1, "j_weight", 0)
        WeightNorm.apply(self.conv1, "k_weight", 0)
        
        self.chomp1 = QChomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)
 
        self.conv2 = QuaternionConv(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilatation=dilation, operation='convolution1d', bias=True)
        
        WeightNorm.apply(self.conv2, "r_weight", 0)
        WeightNorm.apply(self.conv2, "i_weight", 0)
        WeightNorm.apply(self.conv2, "j_weight", 0)
        WeightNorm.apply(self.conv2, "k_weight", 0)
        self.chomp2 = QChomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,  self.dropout1, 
                                 self.conv2,  self.chomp2,  self.relu2, self.dropout2
                                 )
        self.downsample = QuaternionConv(n_inputs, n_outputs, 1, stride=1, operation='convolution1d', bias=False) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()

    def forward(self, x):
        
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class QTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(QTemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        dilation_size = 1
        in_channels = num_inputs
        out_channels = num_channels[0]

        layers += [QTemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        for i in range(1, num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            if i % 1 == 0:
                print("QBLOCK")
                layers += [QTemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout) ]
            else:
                print("BLOCK")
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
