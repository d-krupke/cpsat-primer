# Declarative Modeling with CP-SAT

A companion to the
[CP-SAT Primer](https://github.com/d-krupke/cpsat-primer). The Primer is the
broader and more extensive resource, covering CP-SAT in depth, including much of
the modeling material treated here and a great deal more. This manuscript is
deliberately narrower: it concentrates on the discipline of _modeling_, how to
turn an informal combinatorial optimization problem into a formulation that is
correct before it is ever handed to the solver. It overlaps with the Primer on
purpose, so that it can be read on its own as a self-contained treatment rather
than requiring the Primer first.

It is written for computer scientists and software engineers who are comfortable
with Python and basic discrete mathematics and want a systematic modeling
workflow rather than ad hoc trial and error.

The manuscript builds the model in stages: from the paradigm shift of describing
a problem instead of writing an algorithm, through mathematical notation as a
working tool, to a paper model, an optional Python _verifier_ that judges
candidate solutions independently of any solver, and finally a working CP-SAT
encoding. It also includes a compact reference to CP-SAT's variable and
constraint vocabulary. The focus is on getting a formulation _correct_ — one a
verifier can check and a solver can execute; performance engineering of models
is deliberately out of scope.

The compiled
[`cpsat_declarative_modeling.pdf`](cpsat_declarative_modeling.pdf) is included in
this folder.

## Building

Requires a XeLaTeX toolchain (TeX Live) with the `doclicense` package.

```sh
make           # build cpsat_declarative_modeling.pdf
make clean     # remove LaTeX aux files
make distclean # clean + remove the PDF
```

## Layout

| Path                             | Contents                                            |
| -------------------------------- | --------------------------------------------------- |
| `cpsat_declarative_modeling.tex` | Main document (preamble, title page, abstract).     |
| `chapters/`                      | One file per chapter of the body.                   |
| `.assets/`                       | Title-page/callout styling and the platypus icons.  |
| `cover.png`                      | Cover illustration.                                 |

## License

© Dominik Krupke. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
