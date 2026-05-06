import torch
import torch.nn as nn
import torch.nn.functional as F



class ABMIL(nn.Module):
    def __init__(self,input_size=512):
        super(ABMIL, self).__init__()
        self.L = input_size
        self.D = 128
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
        )
        self.lr = 2e-4

    def forward(self, x, return_WSI_attn=False):
        x = x.squeeze(0)
        forward_return = {}

        A = self.attention(x)  # NxK
        A = torch.transpose(A, -1, -2)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.matmul(A, x)  # KxL

        Y_prob = self.classifier(M)
        Y_prob = F.sigmoid(Y_prob)

        forward_return["logits"] = Y_prob.squeeze()
        if return_WSI_attn:
            forward_return["A"] = A

        return forward_return


    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A



class mcabmil(nn.Module):
    def __init__(self,input_size=512, output_size=1):
        super(mcabmil, self).__init__()
        self.L = input_size
        self.D = 128
        self.K = 1
        self.output_size=output_size

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.output_size)
        )

    def forward(self, x):
        # x = x.squeeze(0)

        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, x)  # KxL

        Y_prob = self.classifier(M)
        
        return Y_prob, A



class gabmil(nn.Module):
    def __init__(self,input_size=512):
        super(gabmil, self).__init__()
        self.input_size = input_size
        self.L = 512
        self.D = 128
        self.K = 1

        self.lr = 5e-4

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh() 
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

        self.dimreduction = DimReduction(self.input_size, self.L)

    def forward(self, x):
        forward_return = {}

        x = x.squeeze(0)

        H = self.dimreduction(x)

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        forward_return["logits"] = Y_prob.squeeze()
        forward_return["A"] = A
        return forward_return

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x