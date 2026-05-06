import os

from argparse import ArgumentParser

from datetime import datetime

import torch
import torch.nn as nn

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from models.instance_based.attenmil import attenmil
from models.instance_based.automil import automil
from models.instance_based.lnpmil import lnpmil
from models.instance_based.mixmil import mixmil

from models.embedding_based.abmil import ABMIL
from models.embedding_based.ac_mil import AC_MIL
from models.embedding_based.amd_mil import AMD_MIL
from models.embedding_based.clam_sb import CLAM_SB_MIL
from models.embedding_based.transmil import TransMIL

from models.litmodels import LitEmbeddingBasedModel, LitInstanceBasedModel

from dataset import WSI_Dataset


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.set_float32_matmul_precision('high')
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(0)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def crossval(name):

    now = datetime.now().strftime("%m_%d")
    B=1

    for i in range(5):

        train_set = WSI_Dataset(f"path/to/training/set")
        val_set = WSI_Dataset(f"path/to/validation/set")

        feature_dim = 512

        if name == "automil":
            model = automil(input_size=feature_dim, output_size=1).cuda()

        if name == "attenmil":
            model = attenmil(input_size=feature_dim, output_size=1).cuda()

        if name == "lnpmil":
            model = lnpmil(input_size=feature_dim, output_size=1).cuda()

        if name == "mixmil":
            model = mixmil(input_size=feature_dim, output_size=1).cuda()

        if name == "ABMIL":
            model = ABMIL(input_size=feature_dim)

        if name == "AC_MIL":
            model = AC_MIL(in_dim=feature_dim, hidden_dim=int(feature_dim/4), num_classes=1).cuda()

        if name == "AMD_MIL":
            model = AMD_MIL(num_classes=1, in_dim=feature_dim, embed_dim=int(feature_dim/4), dropout=0.1, act = nn.ReLU()).cuda()

        if name == "TransMIL":
            model = TransMIL(input_size=feature_dim, n_classes=1).cuda()

        if name == "CLAM_SB_MIL":
            model = CLAM_SB_MIL(in_dim=feature_dim).cuda()


        if name in ["automil", "attenmil", "lnpmil", "mixmil"]:
            litmodel = LitInstanceBasedModel(model)

        elif name in ["ABMIL", "AC_MIL", "AMD_MIL", "TransMIL", "CLAM_SB_MIL"]:
            litmodel = LitEmbeddingBasedModel(model)
        
        else:
            raise ValueError('Model name must be one of the following: "automil", "attenmil", "lnpmil", "mixmil", "AC_MIL", "AMD_MIL", "TransMIL", "CLAM_SB_MIL", "ABMIL"')


        checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="val_loss", mode='min', dirpath=f"proper_{now}_dim_{feature_dim}/checkpoints/{name}/model_{i}", filename="{epoch:02d}-{val_loss}", save_last=True)

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=B, num_workers=12)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=B, num_workers=12)

        logger = TensorBoardLogger(f"proper_{now}_dim_{feature_dim}/lightning_logs", name=name)

        trainer = L.Trainer(max_epochs=200, check_val_every_n_epoch=1, callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode='min', patience=30)], logger=logger)
      
        trainer.fit(litmodel, train_dataloader, val_dataloader)

        with open(f'proper_{now}_dim_{feature_dim}/best_models.txt', 'a') as f:
            f.write(checkpoint_callback.best_model_path)
            f.write('\n')


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    
    args = parser.parse_args()
    name = args.model_name
    
    crossval(name)


if __name__  == "__main__":
    main()
