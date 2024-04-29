<!-- EDIT THIS PART VIA 03_big_picture.md -->

<a name="section-alternatives"></a> <a name="03-big-picture"></a>

## Alternatives: CP-SAT's Place in the World of Optimization

When you begin exploring optimization, you will encounter a plethora of tools,
techniques, and communities. It can be overwhelming, as these groups, while
sharing many similarities, also differ significantly. They might use the same
terminology for different concepts or different terms for the same concepts,
adding to the confusion. Surprisingly, only a handful of experts can effectively
navigate these varied communities and techniques. Often, even specialists,
including professors, concentrate on a singular technique or community,
remaining unaware of potentially more efficient methods developed in other
circles.

If you are interested in understanding the connections between different
optimization concepts, consider watching the talk
[Logic, Optimization, and Constraint Programming: A Fruitful Collaboration](https://simons.berkeley.edu/talks/john-hooker-carnegie-mellon-university-2023-04-19)
by John Hooker. Please note that this is an academic presentation and assumes
some familiarity with theoretical computer science.

Let us now explore the various tools and techniques available, and see how they
compare to CP-SAT. Please note that this overview is high-level and not
exhaustive. If you have any significant additions, feel free to open an issue,
and I will consider including them.

- **Mixed Integer (Linear) Programming (MIP):** MIP is a highly effective method
  for solving a variety of optimization problems, particularly those involving
  networks like flow or tour problems. While MIP only supports linear
  constraints, making it less expressive than CP-SAT, it is often the best
  choice if your model is compatible with these constraints. CP-SAT incorporates
  techniques from MIP, but with limitations, including the absence of continuous
  variables and incremental modeling. Consequently, pure MIP-solvers, being more
  specialized, tend to offer greater efficiency for certain applications.
  Notable MIP-solvers include:
  - [Gurobi](https://www.gurobi.com/): A commercial solver known for its
    state-of-the-art capabilities in MIP-solving. It offers free academic
    licenses, exceptional performance, user-friendliness, and comprehensive
    support through documentation and expert-led webinars. Gurobi is
    particularly impressive in handling complex, large-scale problems.
  - [SCIP](https://www.scipopt.org/): An open-source solver that provides a
    Python interface. Although not as efficient or user-friendly as Gurobi, SCIP
    allows extensive customization and is ideal for research and development,
    especially for experts needing to implement advanced decomposition
    techniques.
  - [HiGHS](https://highs.dev/): A newer solver licensed under MIT, presenting
    an interesting alternative to SCIP. It is possibly faster and features a
    more user-friendly interface, but is less versatile. For performance
    benchmarks, see [here](https://plato.asu.edu/ftp/milp.html).
- **Constraint Programming (CP):** Constraint Programming is a more general
  approach to optimization problems than MIP. As the name suggests, it focuses
  on constraints and solvers usually come with a lot of advanced constraints
  that can be used to describe your problem more naturally. A classical example
  is the `AllDifferent`-constraint, which is very hard to model in MIP, but
  would allow, e.g., to trivially model a Sudoku problem. Constraint Programming
  has been very successful for example in solving scheduling problems, where you
  have a lot of constraints that are hard to model with linear constraints. The
  internal techniques of CP-solvers are often more logic-based and less linear
  algebra-based than MIP-solvers. Popular CP-solvers are:
  - [OR-Tools' CP-SAT](https://github.com/google/or-tools/): This is the solver
    we are talking about in this primer. It internally uses a lot of different
    techniques, including techniques from MIP-solvers, but its primary technique
    is Lazy Clause Generation, which internally translates the problem into a
    SAT-formula and then uses a SAT-solver to solve it.
  - [Choco](https://choco-solver.org/): Choco is a classical constraint
    programming solver in Java. It is under BSD 4-Clause license and comes with
    a lot of features. It is probably not as efficient or modern as CP-SAT, but
    it has some nice features such as the ability to add your own propagators.
- **Logic Programming:** Having mentioned constraint programming, we should also
  mention logic programming. Logic programming is a more generic approach to
  constraint programming, actually being a full turing-complete programming
  language. A popular logic programming language is
  [Prolog](https://en.wikipedia.org/wiki/Prolog). While they may be the final
  solution for solving combinatorial optimization problems, they are not smart
  enough, yet, and you should go for the less expressive constraint programming
  for now.
- **SAT-Solvers:** If your problem is actually just a SAT-problem, you may want
  to use a SAT-solver. SAT-solvers are surprisingly efficient and can often
  handle problems with millions of variables. If you are clever, you can also do
  some optimization problems with SAT-solvers, as CP-SAT actually does. Most
  SAT-solvers support incremental modelling, and some support cardinality
  constraints. However, they are pretty low-level and CP-SAT actually can
  achieve similar performance for many problems. A popular library for
  SAT-solvers is:
  - [PySAT](https://pysathq.github.io/): PySAT is a Python library under MIT
    license that provides a nice interface to many SAT-solvers. It is very easy
    to use and allows you to switch between different solvers without changing
    your code. It is a good choice if you want to experiment with SAT-solvers.
  - There are many solvers popping up every year and many of them are open
    source. Check out the [SAT Competition](http://www.satcompetition.org/) to
    see the current state of the art. Most of the solvers are written in C or
    C++ and do not provide much documentation. However, as SAT-formulas are very
    simple and the solvers usually do not have complex dependencies, they can
    still be reasonably easy to use.
- **Satisfiability modulo theories (SMT):** A level above SAT-solvers are
  SMT-solvers whose goal is to check mathematical formulas for satisfiability.
  They essentially extend the propositional logic of SAT-solvers with theories,
  such as linear arithmetic, bit-vectors, arrays, quantors, and more. For
  example, you can ask an SMT-solver to check if a formula is satisfiable under
  the assumption that all variables are integers and satisfy some linear
  constraints. They are often used for automated theorem proofing or
  verification. A popular SMT-solver is:
  - [Z3](https://github.com/z3prover/z3): Z3 is an SMT solver by Microsoft under
    MIT license. It has a nice Python interface and a good documentation.
- **Nonlinear Programming (NLP):** Many MIP-solvers can actually handle some
  nonlinear constraints, as people noticed that some techniques are actually
  more general than just for linear constraints, e.g., interior points methods
  can also solve second-order cone problems. However, you will notice serious
  performance downgrades. If your constraints and objectives get too complex,
  they may also no longer be a viable option. If you have smaller optimization
  problems of (nearly) any kind, you may want to look into:
  - [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html): SciPy is
    a Python library that offers a wide range of optimization algorithms. Do not
    expect it to get anywhere near the performance of a specialized solver, but
    it gives you a bunch of different options to solve a multitude of problems.
- **Modelling Languages:** As you have seen, there are many solvers and
  techniques to solve optimization problems. However, many optimization problems
  in the real world are actually pretty small and easily solvable, but the
  primary challenge is to model them correctly and quickly. Another issue is
  that some optimization problems have to be solved repeatedly over a long
  period, and you do not want to be dependent on a specific solver, as better
  solvers come out frequently, the licenses may change, or your business experts
  simply should be able to focus on the pure modelling and should not need to
  know about the internals of the solver. For these reasons, there are modelling
  languages that allow you to model your problem in a very high-level way and
  then let the system decide which solver to use and how to solve it. For
  example, the solver can use CP-SAT or Gurobi, without you needing to change
  your model. A disadvantage of these modelling languages is that you give up a
  lot of control and flexibility, in the favor of generality and ease of use.
  The most popular modelling languages are:
  - [AMPL](https://ampl.com/): AMPL is possibly the most popular modelling
    language. It has free and commercial solvers. There is not only extensive
    documentation, but even a book on how to use it.
  - [GAMS](https://www.gams.com/): This is a commercial system which supports
    many solvers and also has gotten a Python-interface. I actually know the
    guys from GAMS as they have a location in Braunschweig. They have a lot of
    experience in optimization, but I have never used their software myself.
  - [pyomo](http://www.pyomo.org/): Pyomo is a Python library that allows you to
    model your optimization problem in Python and then let it solve by different
    solvers. It is not as high-level as AMPL or GAMS, but it is free and open
    source. It is also very flexible and allows you to use Python to model your
    problem, which is a huge advantage if you are already familiar with Python.
    It actually has support for CP-SAT and could be an option if you just want
    to have a quick solution.
  - [OR-Tools' MathOpt](https://developers.google.com/optimization/math_opt): A
    very new competitor and sibling of CP-SAT. It only supports a few solvers,
    but may be still interesting.
- **Specialized Algorithms:** For many optimization problems, there are
  specialized algorithms that can be much more efficient than general-purpose
  solvers. Examples are:
  - [Concorde](http://www.math.uwaterloo.ca/tsp/concorde.html): Concorde is a
    solver for the Traveling Salesman Problem that despite its age is still
    blazingly fast for many instances.
  - [OR-Tools' Routing Solver](https://developers.google.com/optimization/routing):
    OR-Tools also comes with a dedicated solver for routing problems.
  - [OR-Tools' Network Flows](https://developers.google.com/optimization/flow):
    OR-Tools also comes with a dedicated solver for network flow problems.
  - ...

As you can see, there are many tools and techniques to solve optimization
problems. CP-SAT is one of the more general approaches and a great option for
many problems. If you frequently have to deal with combinatorial optimization
problems, you should definitely get familiar with CP-SAT. It is good enough for
most problems, and superior for some, which is amazing for a free and
open-source tool.

---
