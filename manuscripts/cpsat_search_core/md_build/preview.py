#!/usr/bin/env python3
"""
Open a fully self-contained local preview of the generated chapter.

The mdbook-built page references the chapter's images by their absolute
raw.githubusercontent URLs, which only resolve once the images are committed and
pushed to `main`. For a local preview we inline those images as data URIs so the
page renders completely offline (only MathJax still loads from its CDN), then
open it in the default browser.

Run *after* `make primer` (which produces .mdbook/book/search_core.html).
"""

from __future__ import annotations

import base64
import re
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUILT = REPO / ".mdbook" / "book" / "search_core.html"
PREVIEW = REPO / ".mdbook" / "book" / "search_core_preview.html"
IMG_PREFIX = "https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/"
MIME = {
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".gif": "image/gif",
}


def main() -> None:
    if not BUILT.exists():
        raise SystemExit(
            f"{BUILT} not found -- run `make primer` (build.py + mdbook build) first."
        )
    html = BUILT.read_text()
    images_dir = (REPO / "images").resolve()

    def inline(m: re.Match) -> str:
        # The capture allows '.' and '/', so resolve the path and refuse to
        # inline anything that escapes images/ (e.g. a "../.." traversal).
        f = (images_dir / m.group(1)).resolve()
        if not f.is_relative_to(images_dir) or not f.is_file():
            return m.group(0)
        data = base64.b64encode(f.read_bytes()).decode()
        return f"data:{MIME.get(f.suffix, 'application/octet-stream')};base64,{data}"

    html, n = re.subn(re.escape(IMG_PREFIX) + r"([A-Za-z0-9_./-]+)", inline, html)
    PREVIEW.write_text(html)
    print(f"inlined {n} images -> {PREVIEW}")
    webbrowser.open(PREVIEW.as_uri())


if __name__ == "__main__":
    main()
