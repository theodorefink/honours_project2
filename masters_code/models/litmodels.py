import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC


class LitInstanceBasedModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        z, _ = self.model(x)

        loss = F.binary_cross_entropy(z, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        z, _ = self.model(x)

        val_loss = F.binary_cross_entropy(z, y)
        self.log("val_loss", val_loss)

        # metric = BinaryAUROC(thresholds=None)
        # auroc = metric(z, y)

        # self.log("val_auroc", auroc)


    def test_step(self, batch, batch_idx):
        x, y, _ = batch

        z, _ = self.model(x)

        metric = BinaryAUROC(thresholds=None)
        auroc = metric(z, y)

        self.log("test_auroc", auroc)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model.lr, weight_decay=1e-5)
        return optimizer



class LitEmbeddingBasedModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        z = self.model(x)["logits"]

        loss = F.binary_cross_entropy(z, y)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        z = self.model(x)["logits"]

        val_loss = F.binary_cross_entropy(z, y)
        self.log("val_loss", val_loss)

        # metric = BinaryAUROC(thresholds=None)
        # auroc = metric(z, y)
        # self.log("val_auroc", auroc)


    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        z, _ = self.model(x)

        metric = BinaryAUROC(thresholds=None)
        auroc = metric(z, y)
        self.log("val_auroc", auroc)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model.lr, weight_decay=1e-5)
        return optimizer

