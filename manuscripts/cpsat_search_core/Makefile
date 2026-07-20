# Makefile for "How CP-SAT Works: The Search Core"
#
# Targets:
#   make            -> build cpsat_search_core.pdf (xelatex/biber/xelatex/xelatex)
#   make clean      -> remove LaTeX aux files
#   make distclean  -> clean + remove the PDF

LATEX      := xelatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
BIBER      := biber
MAIN       := cpsat_search_core
TEX_SRC    := $(MAIN).tex
BIB_SRC    := references.bib
PDF        := $(MAIN).pdf

FIGURES := $(wildcard figures/*)

AUX_EXT := aux log out toc lof lot fls fdb_latexmk synctex.gz \
           bbl bcf blg run.xml

.PHONY: all build clean distclean

all: build

build: $(PDF)

$(PDF): $(TEX_SRC) $(BIB_SRC) $(FIGURES)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)
	$(BIBER) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)
	$(LATEX) $(LATEXFLAGS) $(TEX_SRC)

clean:
	$(foreach ext,$(AUX_EXT),rm -f $(MAIN).$(ext);)

distclean: clean
	rm -f $(PDF)
