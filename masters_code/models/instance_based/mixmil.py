import torch
import torch.nn as nn
import torch.nn.functional as F



# Found at https://github.com/mammadov7/wsi_classification_pipeline/blob/main/train/mils.py


class mixmil(nn.Module):
    def __init__(self, input_size, output_size, alpha=0.5):
        super(mixmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size    
        self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.lr=1e-4

    def forward(self, x):
        Y_probs = self.classifier(x)
        sig_alpha = torch.sigmoid(self.alpha)
        Y_prob = sig_alpha * torch.amax(Y_probs) + (1 - sig_alpha) * torch.mean(Y_probs)  
        return F.sigmoid(Y_prob.squeeze()), F.softmax(Y_probs, dim=1) 


