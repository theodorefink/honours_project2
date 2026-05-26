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

from conch.open_clip_custom import create_model_from_pretrained
from models.embedding_based.amd_mil import AMD_MIL
from models.litmodels import LitEmbeddingBasedModel
from dataset import WSI_Dataset

IMPORTANCE_THRESHOLD = 0.1
TILE_SIZE = 224


def load_models(ckpt_path, feature_dim=512):
    amd_model = AMD_MIL(num_classes=1, in_dim=feature_dim, embed_dim=feature_dim//4, dropout=0., act=nn.ReLU()).cuda()
    litmodel = LitEmbeddingBasedModel.load_from_checkpoint(ckpt_path, model=amd_model)
    litmodel.eval()

    conch_model, transform = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/CONCH')
    conch_model = conch_model.cuda()
    conch_model.eval()

    return litmodel, conch_model, transform


def get_tile_gradcam(litmodel, features, coords):
    feats = features.cuda().float().unsqueeze(0)
    feats.requires_grad_(True)

    output = litmodel.model(feats)
    pred = output["logits"]
    litmodel.model.zero_grad()
    pred.backward()

    embedding_grads = feats.grad.squeeze(0)
    tile_importance = torch.relu(torch.mean(embedding_grads, dim=-1))

    if tile_importance.max() > tile_importance.min():
        tile_importance = (tile_importance - tile_importance.min()) / (tile_importance.max() - tile_importance.min())

    return tile_importance.detach(), embedding_grads.detach(), pred.detach()


def get_pixel_gradcam(conch_model, transform, slide, coord, embedding_grad, slide_mpp=0.2446):
    # Convert µm coords to pixel coords
    x = int(coord[0].item() / slide_mpp)
    y = int(coord[1].item() / slide_mpp)
    
    try:
        tile = slide.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE)).convert('RGB')
    except Exception as e:
        print(f"  Error reading tile at ({x},{y}): {e}")
        return None

    tile_tensor = transform(tile).unsqueeze(0).cuda()
    tile_tensor.requires_grad_(True)

    embedding = conch_model.encode_image(tile_tensor, normalize=False)

    embedding_grad_cuda = embedding_grad.cuda()
    scalar = (embedding * embedding_grad_cuda).sum()
    scalar.backward()

    pixel_grads = tile_tensor.grad.squeeze(0)
    pixel_importance = pixel_grads.abs().max(dim=0).values
    pixel_importance = torch.relu(pixel_importance)

    return pixel_importance.detach().cpu().numpy(), np.array(tile)


def process_slide(slide, coords, tile_importance, embedding_grads,
                  conch_model, transform, pred, label, path, base_out_dir, top_k=5, slide_mpp=0.2446):
    
    slide_name = Path(path).stem
    true_label = "positive" if label == 1 else "negative"
    
    # Each slide gets its own directory
    slide_dir = os.path.join(base_out_dir, f"pixel_level_{slide_name}")
    os.makedirs(slide_dir, exist_ok=True)

    # --- Full slide overview ---
    slide_mpp = 0.2446
    dims_um = np.array(slide.dimensions) * slide_mpp
    thumb_size = np.round(dims_um * 8 / 256).astype(int)
    thumb = np.array(slide.get_thumbnail(thumb_size))

    scale_x = thumb.shape[1] / slide.dimensions[0]
    scale_y = thumb.shape[0] / slide.dimensions[1]

    pixel_map = np.zeros((thumb.shape[0], thumb.shape[1]), dtype=np.float32)

    important_tiles = (tile_importance > IMPORTANCE_THRESHOLD).nonzero().squeeze(-1)
    print(f"  {slide_name}: {len(important_tiles)} important tiles (threshold={IMPORTANCE_THRESHOLD})")

    # Save original thumbnail
    Image.fromarray(thumb).save(os.path.join(slide_dir, "original_thumbnail.png"))
    print(f"  Saved original thumbnail for {slide_name}")

    # Store pixel results for reuse in zoomed tiles
    tile_pixel_results = {}

    for tile_idx in important_tiles:
        tile_idx = tile_idx.item()
        coord = coords[tile_idx]
        emb_grad = embedding_grads[tile_idx]

        result = get_pixel_gradcam(conch_model, transform, slide, coord, emb_grad)
        if result is None:
            continue

        tile_pixel_results[tile_idx] = result
        pixel_imp, _ = result

        x, y = int(coord[0].item() / slide_mpp), int(coord[1].item() / slide_mpp)
        tx_start = int(x * scale_x)
        ty_start = int(y * scale_y)
        tx_end = min(tx_start + int(TILE_SIZE * scale_x) + 1, thumb.shape[1])
        ty_end = min(ty_start + int(TILE_SIZE * scale_y) + 1, thumb.shape[0])

        tile_w = tx_end - tx_start
        tile_h = ty_end - ty_start
        if tile_w <= 0 or tile_h <= 0:
            continue

        imp_resized = np.array(Image.fromarray(pixel_imp).resize((tile_w, tile_h), Image.BILINEAR))
        pixel_map[ty_start:ty_end, tx_start:tx_end] = np.maximum(
            pixel_map[ty_start:ty_end, tx_start:tx_end], imp_resized)

    if pixel_map.max() > 0:
        pixel_map = pixel_map / pixel_map.max()

    score_im = plt.get_cmap("RdBu_r")(pixel_map)
    score_rgba = (score_im * 255).astype(np.uint8)
    thumb_float = thumb.astype(float) / 255.0
    score_float = score_rgba[:, :, :3].astype(float) / 255.0

    alpha = 0.5
    mask = pixel_map > 0
    overlay = thumb_float.copy()
    overlay[mask] = alpha * score_float[mask] + (1 - alpha) * thumb_float[mask]
    overlay = (overlay * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(overlay)
    ax.set_title(f"{slide_name} | true={true_label} | pred={pred.item():.3f}\nPixel-level GradCAM (threshold={IMPORTANCE_THRESHOLD})", fontsize=13)
    ax.axis("off")
    legend_elements = [
        matplotlib.patches.Patch(facecolor='red', alpha=alpha, label='High importance'),
        matplotlib.patches.Patch(facecolor='blue', alpha=alpha, label='Low importance')
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(slide_dir, "overview.png"), dpi=150)
    plt.close()
    print(f"  Saved overview for {slide_name}")

    # --- Zoomed tiles ---
    top_indices = tile_importance.argsort(descending=True)[:top_k]

    for rank, tile_idx in enumerate(top_indices):
        tile_idx = tile_idx.item()
        importance_score = tile_importance[tile_idx].item()

        if importance_score < IMPORTANCE_THRESHOLD:
            break

        # Reuse already computed result if available
        if tile_idx in tile_pixel_results:
            result = tile_pixel_results[tile_idx]
        else:
            coord = coords[tile_idx]
            emb_grad = embedding_grads[tile_idx]
            result = get_pixel_gradcam(conch_model, transform, slide, coord, emb_grad)

        if result is None:
            continue

        pixel_imp, tile_rgb = result

        if pixel_imp.max() > 0:
            pixel_imp = pixel_imp / pixel_imp.max()

        score_im = plt.get_cmap("RdBu_r")(pixel_imp)
        score_rgba = (score_im * 255).astype(np.uint8)
        # Ensure score matches tile_rgb dimensions
        score_rgba = np.array(Image.fromarray(score_rgba[:,:,:3]).resize(
            (tile_rgb.shape[1], tile_rgb.shape[0]), Image.BILINEAR))
        thumb_float = tile_rgb.astype(float) / 255.0
        score_float = score_rgba.astype(float) / 255.0
        alpha = 0.5
        overlay = alpha * score_float + (1 - alpha) * thumb_float
        overlay = (overlay * 255).astype(np.uint8)

        coord = coords[tile_idx]
        x, y = int(coord[0].item() / slide_mpp), int(coord[1].item() / slide_mpp)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(tile_rgb)
        axes[0].set_title("Original tile", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Pixel GradCAM (importance={importance_score:.3f})", fontsize=12)
        axes[1].axis("off")

        fig.suptitle(f"{slide_name} | tile rank {rank+1} | coord=({x},{y})", fontsize=11)
        legend_elements = [
            matplotlib.patches.Patch(facecolor='red', alpha=alpha, label='High importance'),
            matplotlib.patches.Patch(facecolor='blue', alpha=alpha, label='Low importance')
        ]
        axes[1].legend(handles=legend_elements, loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(slide_dir, f"rank{rank+1}_tile_{x}_{y}.png"), dpi=200)
        plt.close()
        print(f"  Saved zoomed tile rank {rank+1} for {slide_name}")


def main():
    ckpt_path = "proper_05_07_dim_512/checkpoints/AMD_MIL/model_3/epoch=00-val_loss=0.2737697660923004.ckpt"
    dataset = WSI_Dataset("../splits/test")
    base_out_dir = "heatmaps/pixel_gradcam"

    slides = [
        (136, "/local/tfi38/Malignant_Polyps/De-identified_POSITIVE_LNM_Cases/LNM_A0385.svs", 1),
        (115, "/local/tfi38/Malignant_Polyps/De-identified_POSITIVE_LNM_Cases/LNM_A0380.svs", 1),
        (30,  "/local/tfi38/Malignant_Polyps/Waikato_Dataset/De_Identified_NEGATIVE_LNM_Cases/LNM_W0111.svs", 0),
        (141, "/media/tfi38/T7 Shield/Canterbury_Dataset/De_Identified_NEGATIVE_LNM_Cases/LNM_C0141.svs", 0),
    ]

    print("Loading models...")
    litmodel, conch_model, transform = load_models(ckpt_path)

    for idx, path, label in slides:
        print(f"\nProcessing {Path(path).stem}...")
        features, _, coords = dataset[idx]

        tile_importance, embedding_grads, pred = get_tile_gradcam(litmodel, features, coords)

        slide = openslide.open_slide(path)
        process_slide(slide, coords, tile_importance, embedding_grads,
                     conch_model, transform, pred, label, path, base_out_dir, top_k=5)
        slide.close()

    print("\nDone!")


if __name__ == "__main__":
    main()