<!-- EDIT THIS PART VIA 00_intro.md -->

# The CP-SAT Primer: Using and Understanding Google OR-Tools' CP-SAT Solver

<!-- START_SKIP_FOR_README -->

![Cover Image](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_1.webp)

<!-- STOP_SKIP_FOR_README -->

_By [Dominik Krupke](https://krupke.cc), TU Braunschweig_

Many [combinatorially difficult](https://en.wikipedia.org/wiki/NP-hardness)
optimization problems can, despite their proven theoretical hardness, be solved
reasonably well in practice. The most successful approach is to use
[Mixed Integer Linear Programming](https://en.wikipedia.org/wiki/Integer_programming)
(MIP) to model the problem and then use a solver to find a solution. The most
successful solvers for MIPs are, e.g., [Gurobi](https://www.gurobi.com/) and
[CPLEX](https://www.ibm.com/analytics/cplex-optimizer), which are both
commercial and expensive (though, free for academics). There are also some open
source solvers, but they are often not as powerful as the commercial ones.
However, even when investing in such a solver, the underlying techniques
([Branch and Bound](https://en.wikipedia.org/wiki/Branch_and_bound) &
[Cut](https://en.wikipedia.org/wiki/Branch_and_cut) on
[Linear Relaxations](https://en.wikipedia.org/wiki/Linear_programming_relaxation))
struggle with some optimization problems, especially if the problem contains a
lot of logical constraints that a solution has to satisfy. In this case, the
[Constraint Programming](https://en.wikipedia.org/wiki/Constraint_programming)
(CP) approach may be more successful. For Constraint Programming, there are many
open source solvers, but they usually do not scale as well as MIP-solvers and
are worse in optimizing objective functions. While MIP-solvers are frequently
able to optimize problems with hundreds of thousands of variables and
constraints, the classical CP-solvers often struggle with problems with more
than a few thousand variables and constraints. However, the relatively new
[CP-SAT](https://developers.google.com/optimization/cp/cp_solver) of Google's
[OR-Tools](https://github.com/google/or-tools/) suite shows to overcome many of
the weaknesses and provides a viable alternative to MIP-solvers, being
competitive for many problems and sometimes even superior.

This unofficial primer shall help you use and understand this powerful tool,
especially if you are coming from the
[Mixed Integer Linear Programming](https://en.wikipedia.org/wiki/Integer_programming)
-community, as it may prove useful in cases where Branch and Bound performs
poorly.

If you are new to combinatorial optimization, I recommend starting with the free
course on Coursera,
[Discrete Optimization](https://www.coursera.org/learn/discrete-optimization)
taught by Pascal Van Hentenryck and Carleton Coffrin. This course provides a
comprehensive introduction in a concise format.

For an engaging exploration of a classic problem within this domain,
[In Pursuit of the Traveling Salesman by Bill Cook](https://press.princeton.edu/books/paperback/9780691163529/in-pursuit-of-the-traveling-salesman)
is highly recommended. This book, along with this
[YouTube talk](https://www.youtube.com/watch?v=5VjphFYQKj8) by the author that
lasts about an hour, offers a practical case study of the well-known Traveling
Salesman Problem. It not only introduces fundamental techniques but also delves
into the community and historical context of the field.

Additionally, the article
[Mathematical Programming](https://www.gurobi.com/resources/math-programming-modeling-basics/)
by CP-SAT's competitor Gurobi offers an insightful introduction to mathematical
programming and modeling. In this context, the term "Programming" does not refer
to coding; rather, it originates from an earlier usage of the word "program",
which denoted a plan of action or a schedule. If this distinction is new to you,
it is a strong indication that you would benefit from reading this article.

For those who prefer a hands-on approach, you can begin with the primer outlined
below and consult the recommended resources as needed.

Please note, while this primer strives to be self-contained, it is designed for
computer science students who already have a basic understanding of
combinatorial optimization as well as experience in coding with Python. Thus,
some prior knowledge in the field might be necessary to fully benefit from the
content provided. If you are a computer science student **without** prior
knowledge in combinatorial optimization, you may find
[this hands-on tutorial](https://pganalyze.com/blog/a-practical-introduction-to-constraint-programming-using-cp-sat)
a little easier to get started with CP-SAT, and come back once you want to dive
deeper.

**Content:**

1. [Installation](#01-installation): Quick installation guide.
2. [Example](#02-example): A short example, showing the usage of CP-SAT.
3. [Basic Modeling](#04-modelling): An overview of variables, objectives, and
   constraints.
4. [Advanced Modeling](#04B-advanced-modelling): More complex constraints, such
   as circuit constraints and intervals.
5. [Parameters](#05-parameters): How to specify CP-SATs behavior, if needed.
   Timelimits, hints, assumptions, parallelization, ...
6. [Understanding the Log](#understanding-the-log): How to interpret the log
7. [Coding Patterns](#06-coding-patterns): Basic design patterns for creating
   maintainable algorithms.
8. [How does it work?](#07-under-the-hood): After we know what we can do with
   CP-SAT, we look into how CP-SAT will do all these things.
9. [Alternatives](#03-big-picture): An overview of the different optimization
   techniques and tools available. Putting CP-SAT into context.
10. [Benchmarking your Model](#08-benchmarking): How to benchmark your model and
    how to interpret the results.
11. [Large Neighborhood Search](#09-lns): The use of CP-SAT to create more
    powerful heuristics.

---

> **Target audience:** People (especially my computer science students at TU
> Braunschweig) with some background in
> [integer programming](https://en.wikipedia.org/wiki/Integer_programming)
> /[linear optimization](https://en.wikipedia.org/wiki/Linear_programming), who
> would like to know an actual viable alternative to
> [Branch and Cut](https://en.wikipedia.org/wiki/Branch_and_cut). However, I try
> to make it understandable for anyone interested in
> [combinatorial optimization](https://en.wikipedia.org/wiki/Combinatorial_optimization).

> **About the Main Author:** [Dr. Dominik Krupke](https://krupke.cc) is a
> postdoctoral researcher with the
> [Algorithms Group](https://www.ibr.cs.tu-bs.de/alg) at TU Braunschweig. He
> specializes in practical solutions to NP-hard problems. Initially focused on
> theoretical computer science, he now applies his expertise to solve what was
> once deemed impossible. This primer, first developed as course material for
> his students, has been extended in his spare time to cater to a wider
> audience.

> **Found a mistake?** Please open an issue or a pull request. You can also just
> write me a quick mail to `krupked@gmail.com`.

> **Want to contribute?** If you are interested in contributing, please open an
> issue or email me with a brief description of your proposal. We can then
> discuss the details. I welcome all assistance and am open to expanding the
> content. Contributors to any section or similar input will be recognized as
> coauthors.

> **Want to use/share this content?** This tutorial can be freely used under
> [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Smaller parts can
> even be copied without any acknowledgement for non-commercial, educational
> purposes.

<!-- START_SKIP_FOR_README -->

> **Why are there so many platypuses in the text?** I enjoy incorporating
> elements in my texts that add a light-hearted touch. The choice of the
> platypus is intentional: much like CP-SAT integrates diverse elements from
> various domains, the platypus combines traits from different animals. The
> platypus also symbolizes Australia, home to the development of a key technique
> in CP-SAT - Lazy Clause Generation (LCG).

<!-- STOP_SKIP_FOR_README -->

---
