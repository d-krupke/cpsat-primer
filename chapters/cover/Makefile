# Makefile for the CP-SAT Primer book cover.
#
# Targets:
#   make            build cover.pdf and rasterise it to ../../images/cover.webp
#   make cover.pdf  build only the PDF
#   make clean      remove LaTeX aux files
#   make distclean  clean + remove cover.pdf/cover.png and the generated WebP
#
# Requires: xelatex (TeX Live), ghostscript (`gs`) and `cwebp` (brew install
# webp). On macOS, TeX Live lives in /Library/TeX/texbin -- add it to PATH if
# `xelatex: command not found`.
#
# The WebP is what the book and the README embed, so it is written straight into
# images/ rather than left here to be copied by hand (WebP to match the other
# artwork there, and because it is ~6x smaller than the PNG at this size). Only
# that file is committed; cover.pdf, cover.png and the aux files are ignored.

LATEX      := xelatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
MAIN       := cover
TEX_SRC    := $(MAIN).tex
PDF        := $(MAIN).pdf
ART        := jetpack_platypus.png

# 200 dpi over a 6x9in page => 1200x1800 px: sharp on retina displays at the
# ~340px width the pages embed it at, without a multi-megabyte file.
PNG       := $(MAIN).png
WEBP      := ../../images/cover.webp
PNG_DPI   := 200
WEBP_QUAL := 94

AUX_EXT := aux log out fls fdb_latexmk synctex.gz

.PHONY: all build clean distclean

all: build

build: $(WEBP)

# Two passes: TikZ's `remember picture` needs the page anchors from the first run.
$(PDF): $(TEX_SRC) $(ART)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)

$(PNG): $(PDF)
	gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r$(PNG_DPI) \
	   -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
	   -sOutputFile=$(PNG) $(PDF)

$(WEBP): $(PNG)
	cwebp -q $(WEBP_QUAL) -m 6 $(PNG) -o $(WEBP)

clean:
	$(foreach ext,$(AUX_EXT),rm -f $(MAIN).$(ext);)

distclean: clean
	rm -f $(PDF) $(PNG) $(WEBP)
