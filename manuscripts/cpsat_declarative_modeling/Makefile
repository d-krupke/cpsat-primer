# Makefile for "Declarative Modeling with CP-SAT"
#
# Targets:
#   make            build cpsat_declarative_modeling.pdf (xelatex twice, for ToC)
#   make clean      remove LaTeX aux files
#   make distclean  clean + remove the PDF

LATEX      := xelatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
MAIN       := cpsat_declarative_modeling
TEX_SRC    := $(MAIN).tex
PDF        := $(MAIN).pdf

CHAPTERS := $(wildcard chapters/chapter-*.tex)
ASSETS   := .assets/callouts.tex cover.png

AUX_EXT := aux log out toc lof lot fls fdb_latexmk synctex.gz

.PHONY: all build clean distclean

all: build

build: $(PDF)

$(PDF): $(TEX_SRC) $(CHAPTERS) $(ASSETS)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)

clean:
	$(foreach ext,$(AUX_EXT),rm -f $(MAIN).$(ext);)

distclean: clean
	rm -f $(PDF)
