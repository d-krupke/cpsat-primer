# Makefile for "How CP-SAT Works: The Search Core"
#
# Editing the manuscript? The pipeline to update the primer from it is:
#
#   make markdown   regenerate the primer chapter + images from the LaTeX
#                   (rebuilds the PDF first so \cref numbers match)
#   make primer     ... then rebuild the primer's README.md and .mdbook/*.md
#                   (this is what you commit; pre-commit also runs build.py)
#   make preview    ... then `mdbook build` and open a self-contained preview
#
# Other targets:
#   make            build cpsat_search_core.pdf only
#   make clean      remove LaTeX aux files
#   make distclean  clean + remove the PDF

LATEX      := xelatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
BIBER      := biber
MAIN       := cpsat_search_core
TEX_SRC    := $(MAIN).tex
BIB_SRC    := references.bib
PDF        := $(MAIN).pdf
# the primer repository root (no trailing comment: it would leak into the value)
ROOT       := $(abspath $(CURDIR)/../..)

SECTIONS := $(wildcard sections/*.tex)
FIGURES  := $(wildcard figures/*)

AUX_EXT := aux log out toc lof lot fls fdb_latexmk synctex.gz \
           bbl bcf blg run.xml

.PHONY: all build markdown primer preview clean distclean

all: build

build: $(PDF)

# Rebuilds whenever the main file, any section, the bibliography, or a figure
# changes -- so editing a section actually triggers a fresh PDF (and .aux).
$(PDF): $(TEX_SRC) $(SECTIONS) $(BIB_SRC) $(FIGURES)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)
	$(BIBER) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)

# Stage 1: LaTeX -> primer chapter + images. Depends on the .aux (built with the
# PDF) so the pseudocode/cross-references carry the same numbers as the print PDF.
markdown: $(MAIN).aux
	python3 md_build/build_assets.py
	python3 md_build/latex_to_md.py

$(MAIN).aux: $(PDF)

# Stage 2: regenerate the primer's README.md and .mdbook/*.md from all chapters
# (build.py needs no LaTeX; pre-commit runs it too, so this is mostly for preview).
primer: markdown
	cd $(ROOT) && python3 build.py

# Stage 3: build the website and open a fully self-contained local preview.
preview: primer
	cd $(ROOT)/.mdbook && mdbook build
	python3 md_build/preview.py

clean:
	$(foreach ext,$(AUX_EXT),rm -f $(MAIN).$(ext);)

distclean: clean
	rm -f $(PDF)
