import torch
import torch.nn as nn
import torch.nn.functional as F


# Found at https://github.com/mammadov7/wsi_classification_pipeline/blob/main/train/mils.py


class automil(nn.Module):
    def __init__(self, input_size, output_size, alpha=1.):
        super(automil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        # self.classifier =  nn.Linear(self.input_size, self.output_size)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        # self.register_parameter("alpha", nn.Parameter(torch.ones(1)))
        self.lr=1e-4


    def forward(self, x): 
        Y_probs = self.classifier(x)
        A = F.softmax(torch.mul(Y_probs, self.alpha), dim=1)
        Y_prob = torch.sum(torch.mul(Y_probs, A)).squeeze()
        return F.sigmoid(Y_prob), A

