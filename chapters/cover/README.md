# Book cover

The cover of the primer: `cover.tex` sets the title, subtitle and author band
around the artwork `jetpack_platypus.png`, and `make` renders that to
`../../images/cover.webp` — the file embedded at the top of `chapters/intro.md`,
which is what both the mdBook site and the generated `README.md` show.

This mirrors what the manuscripts under `manuscripts/` do for their title pages,
except that the cover is a standalone one-page document, so it can be re-rendered
without building a whole book.

## Building

```sh
make            # cover.pdf -> cover.png -> ../../images/cover.webp
make clean      # drop the LaTeX aux files
make distclean  # also drop cover.pdf, cover.png and the generated WebP
```

Needs `xelatex` (TeX Live), `gs` (Ghostscript) and `cwebp` (`brew install webp`).
On macOS, TeX Live is in `/Library/TeX/texbin`, which is not always on `PATH`:

```sh
PATH="/Library/TeX/texbin:$PATH" make
```

Only `images/cover.webp` is committed; `cover.pdf` and `cover.png` are build
intermediates, ignored via the repository-root `.gitignore` (nested `.gitignore`
files are themselves ignored in this repo, so the rules have to live at the root).

## Changing the cover

Edit the `\cover...` macros near the top of `cover.tex` for the wording, and the
`\definecolor` block for the palette. The layout itself is one absolutely
positioned TikZ picture with every element placed relative to a page corner, so
elements do not push each other around — if you add or remove a line in the
author band, resize the band and the artwork together, as noted in the source.

Two things in there are easy to break and hard to spot:

- **The page tone is `FDFDFD`, not white.** The artwork's own paper is grey
  252–254; matching it is what makes the square edge of the image invisible.
  Pure white leaves a faint seam around the illustration.
- **The artwork is cropped** (`trim=0 0 200 50`) to cut the empty right margin
  and the empty strip above the beak. Swapping in a different illustration means
  re-measuring that crop.
