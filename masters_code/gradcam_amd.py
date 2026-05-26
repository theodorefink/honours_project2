import os
import numpy as np
import torch
import torch.nn as nn
import openslide
import matplotlib
from pathlib import Path
from typing import cast
from PIL import Image
from matplotlib import pyplot as plt

from models.embedding_based.amd_mil import AMD_MIL
from models.litmodels import LitEmbeddingBasedModel
from dataset import WSI_Dataset


def gradcam_vals(dataset, ckpt_path, idx):
    feature_dim = dataset[0][0].shape[-1]
    model = AMD_MIL(num_classes=1, in_dim=feature_dim, embed_dim=int(feature_dim/4), dropout=0., act=nn.ReLU()).cuda()
    litmodel = LitEmbeddingBasedModel.load_from_checkpoint(ckpt_path, model=model)
    litmodel.eval()

    feats = dataset[idx][0].cuda().float().unsqueeze(0)  # 1 x N x 512
    feats.requires_grad_(True)

    output = litmodel.model(feats)
    pred = output["logits"]

    # backpropagate gradient of output w.r.t. input features
    litmodel.model.zero_grad()
    pred.backward()

    gradients = feats.grad.squeeze(0)  # N x 512

    # GradCAM: ReLU of mean gradient across feature dimension
    importance = torch.mean(gradients, dim=-1)  # N
    importance = torch.relu(importance)

    # normalise
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min())

    return importance.detach(), pred.detach()


def get_stride(coords):
    xs = coords[:, 0].unique(sorted=True)
    ys = coords[:, 1].unique(sorted=True)
    stride = cast(float, min(
        (xs[1:] - xs[:-1]).min().item(),
        (ys[1:] - ys[:-1]).min().item(),
    ))
    return stride


def make_heatmap(importance, path, coords, pred, label, out_dir, suffix="gradcam"):
    stride = get_stride(coords)
    coords_norm = (coords / stride).round().long()

    slide = openslide.open_slide(path)
    slide_mpp = 0.2446
    dims_um = np.array(slide.dimensions) * slide_mpp
    size = coords_norm.max(0).values.flip(0) + 1

    im = torch.zeros(*size.tolist()).type_as(importance)
    flattened_im = im.flatten()
    flattened_coords = coords_norm[:,1] * im.shape[1] + coords_norm[:,0]
    flattened_im[flattened_coords] = importance
    im = flattened_im.reshape_as(im)
    im = im.cpu().detach().numpy()

    score_im = cast(np.ndarray, plt.get_cmap("RdBu_r")(im))
    thumb = np.array(slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int)))
    thumb = thumb[:im.shape[0] * 8, :im.shape[1] * 8]
    thumb_height, thumb_width = thumb.shape[:2]

    score_resized = Image.fromarray(np.uint8(score_im * 255), mode='RGBA').resize(
        (thumb_width, thumb_height), resample=Image.Resampling.NEAREST)
    score_resized = np.array(score_resized) / 255.0
    thumb_float = thumb.astype(float) / 255.0

    alpha = 0.5
    mask = score_resized[..., -1] > 0
    overlay = thumb_float.copy()
    overlay[mask] = alpha * score_resized[mask, :3] + (1 - alpha) * thumb_float[mask]
    overlay = (overlay * 255).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    slide_name = Path(path).stem
    true_label = "positive" if label == 1 else "negative"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(overlay)
    ax.set_title(f"{slide_name} | true={true_label} | pred={pred.item():.3f}", fontsize=14, pad=20)
    ax.axis("off")
    legend_elements = [
        matplotlib.patches.Patch(facecolor='red', alpha=alpha, label='High importance'),
        matplotlib.patches.Patch(facecolor='blue', alpha=alpha, label='Low importance')
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{slide_name}_{suffix}.png", dpi=150)
    plt.close()
    print(f"Saved {suffix} heatmap for {slide_name}")


def main():
    ckpt_path = "proper_05_07_dim_512/checkpoints/AMD_MIL/model_3/epoch=00-val_loss=0.2737697660923004.ckpt"
    dataset = WSI_Dataset("../splits/test")
    out_dir = "heatmaps/gradcam"

    slides = [
        (136, "/local/tfi38/Malignant_Polyps/De-identified_POSITIVE_LNM_Cases/LNM_A0385.svs", 1),
        (115, "/local/tfi38/Malignant_Polyps/De-identified_POSITIVE_LNM_Cases/LNM_A0380.svs", 1),
        (30,  "/local/tfi38/Malignant_Polyps/Waikato_Dataset/De_Identified_NEGATIVE_LNM_Cases/LNM_W0111.svs", 0),
        (141, "/media/tfi38/T7 Shield/Canterbury_Dataset/De_Identified_NEGATIVE_LNM_Cases/LNM_C0141.svs", 0),
    ]

    with open('importance.txt', 'w') as f:
        for idx, path, label in slides:
            importance, pred = gradcam_vals(dataset, ckpt_path, idx)
            importance_list = importance.cpu().tolist()
            f.write(f"{path}\n")
            for i in importance_list:
                f.write(f"{i}\n")
            coords = dataset[idx][2]
            make_heatmap(importance, path, coords, pred, label, out_dir)


if __name__ == "__main__":
    main()