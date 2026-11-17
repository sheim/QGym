"""Study-only figure writer: high dpi first, then fit the repo's 100 KiB limit.

The product repo rejects any committed file above 100 KiB, so a study figure is
rendered at high resolution and, only if that exceeds the limit, palette-reduced
and downscaled until it fits. Line art survives quantisation well; the
alternative (rendering at 100 dpi) is visibly worse.
"""

from __future__ import annotations

import pathlib

from PIL import Image

LIMIT_BYTES = 100 * 1024


def save_figure(
    fig,
    path,
    *,
    dpi: int = 200,
    limit_bytes: int = LIMIT_BYTES,
    layout: bool = True,
):
    """Render ``fig`` to ``path`` at ``dpi``, then shrink the file to fit.

    ``layout=False`` keeps a caller-supplied layout (e.g. ``tight_layout`` with a
    rect reserved for a figure legend); calling it here would undo that.
    """
    path = pathlib.Path(path)
    if layout:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, pil_kwargs={"optimize": True})
    import matplotlib.pyplot as plt

    plt.close(fig)
    if path.stat().st_size <= limit_bytes:
        return _describe(path, dpi)
    original = Image.open(path).convert("RGB")
    # Prefer pixels over palette depth: hold the largest size that fits, and only
    # drop colours when even the smallest palette is too big.
    for scale in (1.0, 0.85, 0.72, 0.6, 0.5, 0.42, 0.35, 0.3):
        width = max(int(original.width * scale), 320)
        height = max(int(original.height * scale), 240)
        resized = original.resize((width, height), Image.LANCZOS)
        for colors in (128, 64, 48, 32, 24, 16, 12):
            resized.convert("P", palette=Image.ADAPTIVE, colors=colors).save(
                path, optimize=True
            )
            if path.stat().st_size <= limit_bytes:
                return _describe(path, dpi, colors=colors, scale=scale)
    return _describe(path, dpi, colors=12, scale=0.3)


def _describe(path: pathlib.Path, dpi: int, colors=None, scale=1.0) -> dict:
    with Image.open(path) as image:
        size = image.size
    info = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "pixels": size,
        "render_dpi": dpi,
        "palette_colors": colors,
        "downscale": scale,
        "effective_dpi": round(dpi * scale, 1),
    }
    print(
        f"figure {path.name}: {size[0]}x{size[1]} px, {info['bytes'] / 1024:.0f} KiB, "
        f"render {dpi} dpi, effective ~{info['effective_dpi']} dpi"
        + (f", {colors} colours" if colors else "")
    )
    return info
