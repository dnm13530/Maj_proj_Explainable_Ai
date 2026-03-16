import numpy as np
import torch
from PIL import Image


class PrettyPrinter:
    """Renders Artifact-Attention Maps as color-coded overlays on input images."""

    def render_overlay(
        self,
        image: Image.Image,
        attention_map: torch.Tensor,
        output_path: str,
        alpha: float = 0.5,
    ) -> None:
        """
        Apply jet colormap to attention map, blend with original image, save to output_path.

        Args:
            image: Original PIL image.
            attention_map: Normalized [H, W] tensor in [0, 1].
            output_path: Path to save the overlay image.
            alpha: Blend factor (0 = original only, 1 = heatmap only).
        """
        import matplotlib.cm as cm

        # Resize attention map to match image size
        img_w, img_h = image.size
        attn_np = attention_map.detach().cpu().numpy()

        # Apply jet colormap → RGBA
        colormap = cm.get_cmap("jet")
        heatmap_rgba = colormap(attn_np)                    # [H, W, 4]
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_rgb).resize((img_w, img_h), Image.BILINEAR)

        # Blend
        original_rgb = image.convert("RGB")
        overlay = Image.blend(original_rgb, heatmap_pil, alpha=alpha)
        overlay.save(output_path)
