import torch
import torch.nn as nn
import torch.nn.functional as F


# Found at https://github.com/mammadov7/wsi_classification_pipeline/blob/main/train/mils.py


class lnpmil(nn.Module):
    def __init__(self, input_size, output_size, norm=1.):
        super(lnpmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.p = nn.Parameter(torch.tensor(norm), requires_grad=True)     
        self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size))
        self.lr=1e-4

    def forward(self, x):
        Y_probs = torch.abs(self.classifier(x))
        norm_p= 1.+torch.log(1.+ torch.exp(self.p))
        Y_prob = F.sigmoid( torch.mean(Y_probs.pow(norm_p)).pow(1. / norm_p)).squeeze()
        return Y_prob, F.softmax(Y_probs, dim=1) 


