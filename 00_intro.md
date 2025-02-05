<!-- EDIT THIS PART VIA 00_intro.md -->

# The CP-SAT Primer: Using and Understanding Google OR-Tools' CP-SAT Solver

<!-- START_SKIP_FOR_README -->

![Cover Image](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_1.webp)

<!-- STOP_SKIP_FOR_README -->

_By [Dominik Krupke](https://krupke.cc), TU Braunschweig, with contributions
from Leon Lan, Michael Perk, and
[others](https://github.com/d-krupke/cpsat-primer/graphs/contributors)._

<!-- Introduction Paragraph --->

Many [combinatorially difficult](https://en.wikipedia.org/wiki/NP-hardness)
optimization problems can, despite their proven theoretical hardness, be solved
reasonably well in practice. The most successful approach is to use
[Mixed Integer Linear Programming](https://en.wikipedia.org/wiki/Integer_programming)
(MIP) to model the problem and then use a solver to find a solution. The most
successful solvers for MIPs are, e.g., [Gurobi](https://www.gurobi.com/),
[CPLEX](https://www.ibm.com/analytics/cplex-optimizer),
[COPT Cardinal Solver](https://www.copt.de/), and
[FICO Xpress Optimization](https://www.fico.com/en/products/fico-xpress-optimization),
which are all commercial and expensive (though, mostly free for academics).
There are also some open source solvers (e.g., [SCIP](https://www.scipopt.org/)
and [HiGHS](https://highs.dev/)), but they are often not as powerful as the
commercial ones (yet). However, even when investing in such a solver, the
underlying techniques
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

As a quick demonstration of CP-SAT's capabilities - particularly for those less
familiar with optimization frameworks - let us solve an instance of the NP-hard
Knapsack Problem. This classic optimization problem requires selecting a subset
of items, each with a specific weight and value, to maximize the total value
without exceeding a weight limit. Although a recursive algorithm is easy to
implement, 100 items yield approximately $2^{100} \approx 10^{30}$ possible
solutions. Even with a supercomputer performing $10^{18}$ operations per second,
it would take more than 31,000 years to evaluate all possibilities.

Here is how you can solve it using CP-SAT:

```python
from ortools.sat.python import cp_model  # pip install -U ortools

# Specifying the input
weights = [395, 658, 113, 185, 336, 494, 294, 295, 256, 530, 311, 321, 602, 855, 209, 647, 520, 387, 743, 26, 54, 420, 667, 971, 171, 354, 962, 454, 589, 131, 342, 449, 648, 14, 201, 150, 602, 831, 941, 747, 444, 982, 732, 350, 683, 279, 667, 400, 441, 786, 309, 887, 189, 119, 209, 532, 461, 420, 14, 788, 691, 510, 961, 528, 538, 476, 49, 404, 761, 435, 729, 245, 204, 401, 347, 674, 75, 40, 882, 520, 692, 104, 512, 97, 713, 779, 224, 357, 193, 431, 442, 816, 920, 28, 143, 388, 23, 374, 905, 942]
values = [71, 15, 100, 37, 77, 28, 71, 30, 40, 22, 28, 39, 43, 61, 57, 100, 28, 47, 32, 66, 79, 70, 86, 86, 22, 57, 29, 38, 83, 73, 91, 54, 61, 63, 45, 30, 51, 5, 83, 18, 72, 89, 27, 66, 43, 64, 22, 23, 22, 72, 10, 29, 59, 45, 65, 38, 22, 68, 23, 13, 45, 34, 63, 34, 38, 30, 82, 33, 64, 100, 26, 50, 66, 40, 85, 71, 54, 25, 100, 74, 96, 62, 58, 21, 35, 36, 91, 7, 19, 32, 77, 70, 23, 43, 78, 98, 30, 12, 76, 38]
capacity = 2000

# Now we solve the problem
model = cp_model.CpModel()
xs = [model.new_bool_var(f"x_{i}") for i in range(len(weights))]

model.add(sum(x * w for x, w in zip(xs, weights)) <= capacity)
model.maximize(sum(x * v for x, v in zip(xs, values)))

solver = cp_model.CpSolver()
solver.solve(model)

print("Optimal selection:", [i for i, x in enumerate(xs) if solver.value(x)])
print("Total packed value:", solver.objective_value)
```

```
Optimal selection: [2, 14, 19, 20, 29, 33, 52, 53, 54, 58, 66, 72, 76, 77, 81, 86, 93, 94, 96]
Total packed value: 1161.0
```

How long did CP-SAT take? On my machine, it found the provably best solution
from $2^{100}$ possibilities in just 0.01 seconds. Feel free to try it on yours.
CP-SAT does not evaluate all solutions; it uses advanced techniques to make
deductions and prune the search space. While more efficient approaches than a
naive recursive algorithm exist, matching CP-SAT’s performance would require
significant time and effort. And this is just the beginning - CP-SAT can tackle
much more complex problems, as we will see in this primer.

> :video:
>
> Not convinced yet of why tools like CP-SAT are amazing? Maybe Marco Lübbecke
> can convince you in his 12-minute TEDx talk
> [Anything you can do I can do better](https://www.youtube.com/watch?v=Dc38La-Xvog)
> about mathematical optimization.

### Content

Whether you are from the MIP community seeking alternatives or CP-SAT is your
first optimization solver, this book will guide you through the fundamentals of
CP-SAT in the first part, demonstrating all its features. The second part will
equip you with the skills needed to build and deploy optimization algorithms
using CP-SAT.

The first part introduces the fundamentals of CP-SAT, starting with a chapter on
installation. This chapter guides you through setting up CP-SAT and outlines the
necessary hardware requirements. The next chapter provides a simple example of
using CP-SAT, explaining the mathematical notation and its approximation in
Python with overloaded operators. You will then progress to basic modeling,
learning how to create variables, objectives, and fundamental constraints in
CP-SAT.

Following this, a chapter on advanced modeling will teach you how to handle
complex constraints, such as circuit constraints and intervals, with practical
examples. Another chapter discusses specifying CP-SAT's behavior, including
setting time limits and using parallelization. You will also find a chapter on
interpreting CP-SAT logs, which helps you understand how well CP-SAT is managing
your problem. Additionally, there is an overview of the underlying techniques
used in CP-SAT. The first part concludes with a chapter comparing CP-SAT with
other optimization techniques and tools, providing a broader context.

The second part delves into more advanced topics, focusing on general skills
like coding patterns and benchmarking rather than specific CP-SAT features. A
chapter on coding patterns offers basic design patterns for creating
maintainable algorithms with CP-SAT. Another chapter explains how to provide
your optimization algorithm as a service by building an optimization API. There
is also a chapter on developing powerful heuristics using CP-SAT for
particularly difficult or large problems. The second part concludes with a
chapter on benchmarking, offering guidance on how to scientifically benchmark
your model and interpret the results.

### Target Audience

I wrote this book for my computer science students at TU Braunschweig, and it is
used as supplementary material in my algorithm engineering courses. Initially,
we focused on Mixed Integer Programming (MIP), with CP-SAT discussed as an
alternative. However, we recently began using CP-SAT as the first optimization
solver due to its high-level interface, which is much easier for beginners to
grasp. Despite this shift, because MIP is more commonly used, the book includes
numerous comparisons to MIP. Thus, it is designed to be beginner-friendly while
also addressing the needs of MIP users seeking alternatives.

Unlike other books aimed at mathematical optimization or operations research
students, this one, aimed at computer science students, emphasizes coding over
mathematics or business cases, providing a hands-on approach to learning
optimization. The second part of the book can also be interesting for more
advanced users, providing content that I found missing in other books on
optimization.

### Table of Content

**Part 1: The Basics**

1. [Installation](#01-installation): Quick installation guide.
2. [Example](#02-example): A short example, showing the usage of CP-SAT.
3. [Basic Modeling](#04-modelling): An overview of variables, objectives, and
   constraints.
4. [Advanced Modeling](#04B-advanced-modelling): More complex constraints, such
   as circuit constraints and intervals.
5. [Parameters](#05-parameters): How to specify CP-SATs behavior, if needed.
   Timelimits, hints, assumptions, parallelization, ...
6. [Understanding the Log](#understanding-the-log): How to interpret the log
7. [How does it work?](#07-under-the-hood): After we know what we can do with
   CP-SAT, we look into how CP-SAT will do all these things.
8. [Alternatives](#03-big-picture): An overview of the different optimization
   techniques and tools available. Putting CP-SAT into context.

**Part 2: Advanced Topics**

7. [Coding Patterns](#06-coding-patterns): Basic design patterns for creating
   maintainable algorithms.
8. [(DRAFT) Building an Optimization API](#building_an_optimization_api) How to
   build a scalable API for long running optimization jobs.
9. [(DRAFT) CP-SAT vs. ML vs. QC](#chapters-machine-learning): A comparison of
   CP-SAT with Machine Learning and Quantum Computing.
10. [Large Neighborhood Search](#09-lns): The use of CP-SAT to create more
    powerful heuristics.
11. [Benchmarking your Model](#08-benchmarking): How to benchmark your model and
    how to interpret the results.

### Background

<!-- Background --->

This book assumes you are fluent in Python, and have already been introduced to
combinatorial optimization problems. In case you are just getting into
combinatorial optimization and are learning on your own, I recommend starting
with the free Coursera course,
[Discrete Optimization](https://www.coursera.org/learn/discrete-optimization),
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

> **About the Lead Author:** [Dr. Dominik Krupke](https://krupke.cc) is a
> postdoctoral researcher with the
> [Algorithms Division](https://www.ibr.cs.tu-bs.de/alg) at TU Braunschweig. He
> specializes in practical solutions to NP-hard problems. Initially focused on
> theoretical computer science, he now applies his expertise to solve what was
> once deemed impossible, frequently with the help of CP-SAT. This primer on
> CP-SAT, first developed as course material for his students, has been extended
> in his spare time to cater to a wider audience.
>
> **Contributors:** This primer has been enriched by the contributions of
> [several individuals](https://github.com/d-krupke/cpsat-primer/graphs/contributors).
> Notably, Leon Lan played a key role in restructuring the content and offering
> critical feedback, while Michael Perk significantly enhanced the section on
> the reservoir constraint. I also extend my gratitude to all other contributors
> who identified and corrected errors, improved the text, and offered valuable
> insights.

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
