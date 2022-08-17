import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRegressor(nn.Module):
    def __init__(self, input_len=14, target_len=4):
        super(SimpleRegressor, self).__init__()

        self.linear1 = nn.Linear(in_features=input_len, out_features=input_len*4)
        self.linear2 = nn.Linear(in_features=input_len*4, out_features=input_len*8)
        self.linear3 = nn.Linear(in_features=input_len*8, out_features=input_len*4)
        self.linear4 = nn.Linear(in_features=input_len*4, out_features=input_len)
        self.linear5 = nn.Linear(in_features=input_len, out_features=target_len)


    def forward(self, x):
        out = self.linear5(F.relu(self.linear4(F.relu(self.linear3(F.relu(self.linear2(F.relu(self.linear1(x)))))))))
        return out
