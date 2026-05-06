import torch
import torch.nn as nn
import torch.nn.functional as F


# Found at https://github.com/mammadov7/wsi_classification_pipeline/blob/main/train/mils.py


class attenmil(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(attenmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier = nn.Linear(self.input_size, self.output_size, nn.Sigmoid())
        # self.classifier =  nn.Linear(self.input_size, self.output_size)      
        self.attention = nn.Linear(self.input_size, 1)
        self.lr = 1e-4

    def forward(self, x):
        Y_probs = self.classifier(x)
        A = F.softmax(self.attention(x), dim=1)
        Y_prob = F.sigmoid(torch.sum(torch.mul(Y_probs, A))).squeeze()
        return Y_prob, A


