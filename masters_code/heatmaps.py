import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import openslide
import matplotlib

from pathlib import Path
from typing import cast
from PIL import Image

from matplotlib import pyplot as plt

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



def attention_vals(dataset, ckpt_root, idx, name):

    with open(f"{ckpt_root}/test_models.txt", 'r') as f:
        lines = f.readlines()

    for line in lines:
        if name in line:
            ckpt_path = line.strip()


    feature_dim = dataset[0][0].shape[-1]

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
        model = AMD_MIL(num_classes=1, in_dim=feature_dim, embed_dim=int(feature_dim/4), dropout=0., act = nn.ReLU()).cuda()
    if name == "TransMIL":
        model = TransMIL(input_size=feature_dim, n_classes=1).cuda()
    if name == "CLAM_SB_MIL":
        model = CLAM_SB_MIL(in_dim=feature_dim).cuda()


    if name in ['automil', 'attenmil', 'lnpmil', 'mixmil']:
        
        litmodel = LitInstanceBasedModel.load_from_checkpoint(ckpt_path, model=model)
        litmodel.eval()

        x = litmodel.model(dataset[idx][0].cuda().float().unsqueeze(0))
        
        classifications = litmodel.model.classifier(dataset[idx][0].cuda().float())

        pred = x[0]
        attn = F.softmax(classifications, dim=0)

    else:
        
        litmodel = LitEmbeddingBasedModel.load_from_checkpoint(ckpt_path, model=model)
        litmodel.eval()

        x = litmodel.model(dataset[idx][0].cuda().float().unsqueeze(0), return_WSI_attn=True)
        pred = x["logits"]
        attn = x["A"].squeeze()


    # no other repo i found actually does this normalisation since the attentions are softmaxed as other repos did, but I found that normalising helped the heatmaps have more distinct colouring, without it the tiles were all very same-y

    attn = (attn-torch.min(attn))/(torch.max(attn)-torch.min(attn))


    return attn, pred

            
def get_stride(coords):
    """Gets the minimum step width between any two coordintes."""
    xs = coords[:, 0].unique(sorted=True)
    ys = coords[:, 1].unique(sorted=True)
    stride = cast(
        float,
        min(
            (xs[1:] - xs[:-1]).min().item(),
            (ys[1:] - ys[:-1]).min().item(),
        ),
    )
    return stride


def make_images(name, attention, path, coords, pred, alpha, ckpt_root, idx):
    """
    this is a piecemeal combination of various functions in the heatmap generation code for STAMP https://github.com/KatherLab/STAMP/tree/main
    """
    
    stride = get_stride(coords)
    coords_norm = (coords / stride).round().long()

    slide = openslide.open_slide(path)
    slide_mpp = 0.2446
    
    dims_um = np.array(slide.dimensions) * slide_mpp
    size = coords_norm.max(0).values.flip(0) + 1

    im = torch.zeros((*size.tolist(), *attention.shape[1:])).type_as(attention)
    flattened_im = im.flatten(end_dim=1)
    flattened_coords = coords_norm[:,1] * im.shape[1] + coords_norm[:,0]
    flattened_im[flattened_coords] = attention
    im = flattened_im.reshape_as(im)
    im = im.squeeze().cpu().detach().numpy()
    score_im = cast(np.ndarray, plt.get_cmap("RdBu_r")(im))

    thumb = np.array(slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int)))
    thumb = thumb[: im.shape[0] * 8, : im.shape[1] * 8]
    thumb_height, thumb_width = thumb.shape[:2]
    
    score_resized = Image.fromarray(np.uint8(score_im * 255), mode='RGBA').resize((thumb_width, thumb_height), resample=Image.Resampling.NEAREST)

    score_resized = np.array(score_resized) / 255.0

    thumb_float = thumb.astype(float) / 255.0

    mask = score_resized[..., -1] > 0
    overlay = thumb_float.copy()

    overlay[mask] = alpha * score_resized[mask, :3] + (1-alpha) * thumb_float[mask]

    overlay = (overlay * 255).astype(np.uint8)

    os.makedirs(f'{ckpt_root}/heatmaps/raw/{idx}', exist_ok=True)
    os.makedirs(f'{ckpt_root}/heatmaps/plotted/{idx}', exist_ok=True)

    
    Image.fromarray(overlay).save(f"{ckpt_root}/heatmaps/raw/{idx}/{name}.png")

    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(overlay)
    ax.set_title(f"Slide Score: {pred}", fontsize=16, pad=20)
    ax.axis("off")

    legend_elements = [
        matplotlib.patches.Patch(facecolor='red', alpha=alpha, label='Positive'),
        matplotlib.patches.Patch(facecolor='blue', alpha=alpha, label='Negative')
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()

    plt.savefig(f"{ckpt_root}/heatmaps/plotted/{idx}/{name}.png")
    plt.close()



def main():
    ckpt_roots = [
        Path("/local/oli24/masters_repo/proper_02_16_dim_512"),
    ]

    dataset = WSI_Dataset(
        "/local/oli24/test_set_conch"
    )

    indices = [
        18,
        122,
        158,
        47,
        23,
        38,
        36,
        144,
        92,
        168,
        6,
        8,
        27
    ]

    paths = [
        "/local/oli24/Datasets/canterbury/positive/LNM_C0008.svs",
        "/local/oli24/Datasets/waikato/negative/LNM_W0141.svs",
        "/local/oli24/Datasets/auckland_dataset/De_identified_NEGATIVE_LNM_Cases/LNM_A0023.svs",
        "/local/oli24/Datasets/auckland_dataset/De_identified_NEGATIVE_LNM_Cases/LNM_A0256.svs",
        "/local/oli24/Datasets/waikato/negative/LNM_W0061.svs",
        "/local/oli24/Datasets/canterbury/negative/LNM_C0222.svs",
        "/local/oli24/Datasets/auckland_dataset/De_identified_NEGATIVE_LNM_Cases/LNM_A0366.svs",
        "/local/oli24/Datasets/waikato/positive/LNM_W0004.svs",
        "/local/oli24/Datasets/auckland_dataset/De-identified_POSITIVE_LNM_Cases/LNM_A0258.svs",
        "/local/oli24/Datasets/canterbury/positive/LNM_C0082.svs",
        "/local/oli24/Datasets/auckland_dataset/De_identified_NEGATIVE_LNM_Cases/LNM_A0014.svs",
        "/local/oli24/Datasets/waikato/negative/LNM_W0120.svs",
        "/local/oli24/Datasets/canterbury/positive/LNM_C0104.svs",
    ]
    
    alpha = 0.5

    for ckpt_root in ckpt_roots:

        names = ['automil', 'attenmil', 'lnpmil', 'mixmil', 'ABMIL', 'AC_MIL', 'AMD_MIL', 'CLAM_SB_MIL', 'TransMIL']

        for name in names:
            for idx, path in zip(indices, paths):
                attention, pred = attention_vals(dataset, ckpt_root, idx, name)
                coords = dataset[idx][2]

                make_images(name, attention, path, coords, pred, alpha, ckpt_root, idx)


if __name__ == '__main__':
    main()

