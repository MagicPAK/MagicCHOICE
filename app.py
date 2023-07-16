import os
import numpy
import torch
from torch import nn

class DecompositionLayer(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AdaptiveAvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        num_of_pads = (self.kernel_size -1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend
    
def autocorrelation(query_states, key_states):
 
    query_states_fft = torch.fft.rfft(query_states, dim=1)
    key_states_fft = torch.fft.rfft(key_states, dim=1)
    attn_weights = query_states_fft * torch.conj(key_states_fft)
    attn_weights = torch.fft.irfft(attn_weights, dim=1)
    
    return attn_weights

attn_output = autocorrelation(2, 1)
print("ok")
