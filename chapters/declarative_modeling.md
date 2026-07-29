## Declarative Modeling with CP-SAT

<a name="chapters-declarative_modeling"></a>

[**Declarative Modeling with CP-SAT**](https://github.com/d-krupke/cpsat-primer/blob/main/manuscripts/cpsat_declarative_modeling/cpsat_declarative_modeling.pdf)
is a standalone companion manuscript to this primer, available as a typeset PDF.

<p align="center">
  <a href="https://github.com/d-krupke/cpsat-primer/blob/main/manuscripts/cpsat_declarative_modeling/cpsat_declarative_modeling.pdf">
    <img src="https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/declarative_modeling_titlepage.png" alt="Declarative Modeling with CP-SAT — title page" width="320">
  </a>
  <br>
  <em><a href="https://github.com/d-krupke/cpsat-primer/blob/main/manuscripts/cpsat_declarative_modeling/cpsat_declarative_modeling.pdf">Click the cover to read the typeset PDF</a></em>
</p>

The primer is the broader, more extensive resource: it covers CP-SAT in depth,
including much of the modeling material and a great deal more. The companion
manuscript is deliberately narrower and approaches the material from a different
angle: the discipline of _modeling_ itself, the path from an informal problem to
a formulation that is correct before it is ever handed to the solver. It
overlaps with the primer on purpose, so that it can be read on its own.

It walks through the modeling workflow end to end: mathematical notation as a
working tool rather than a gatekeeping ritual; the five-part decomposition of a
model into entities, parameters, decision variables, constraints, and an
objective; an optional Python _verifier_ that judges the feasibility and quality
of candidate solutions independently of any solver; and the translation of a
paper model into a working CP-SAT encoding, with a compact reference to CP-SAT's
variable and constraint vocabulary. The focus is on getting a formulation
_correct_; performance engineering is left to the primer.

The
[source and build instructions](https://github.com/d-krupke/cpsat-primer/tree/main/manuscripts/cpsat_declarative_modeling)
live alongside this primer in the `manuscripts/` directory.
