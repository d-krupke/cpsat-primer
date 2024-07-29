<!-- EDIT THIS PART VIA 03_big_picture.md -->

<a name="section-alternatives"></a> <a name="03-big-picture"></a>

## Alternatives: CP-SAT's Place in the World of Optimization

<!-- START_SKIP_FOR_README -->

![Cover Image World of Optimization](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_8.webp)

<!-- STOP_SKIP_FOR_README -->

When you begin exploring optimization, you will encounter a plethora of tools,
techniques, and communities. It can be overwhelming, as these groups, while
sharing many similarities, also differ significantly. They might use the same
terminology for different concepts or different terms for the same concepts,
adding to the confusion. Not too many experts can effectively navigate these
varied communities and techniques. Often, even specialists, including
professors, concentrate on a singular technique or community, remaining unaware
of potentially more efficient methods developed in other circles.

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
  - [OR-Tools' CP-SAT](https://github.com/google/or-tools/): Discussed in this
    primer, CP-SAT combines various optimization techniques, including those
    from MIP solvers, but its primary technique is Lazy Clause Generation. This
    approach translates problems into (Max-)SAT formulas for resolution.
  - [Choco](https://choco-solver.org/): A traditional CP solver developed in
    Java, licensed under the BSD 4-Clause. While it may not match CP-SAT in
    efficiency or modernity, Choco offers significant flexibility, including the
    option to integrate custom propagators.
- **SAT-Solvers:** If your problem is actually just to find a feasible solution
  for some boolean variables, you may want to use a SAT-solver. SAT-solvers are
  surprisingly efficient and can often handle problems with millions of
  variables. If you are clever, you can also do some optimization problems with
  SAT-solvers, as CP-SAT actually does. Most SAT-solvers support incremental
  modelling, and some support cardinality constraints. However, they are pretty
  low-level and CP-SAT actually can achieve similar performance for many
  problems. A popular library for SAT-solvers is:
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
- **Satisfiability modulo theories (SMT):** SMT-solvers represent an advanced
  tier above traditional SAT-solvers. They aim to check the satisfiability of
  mathematical formulas by extending propositional logic with additional
  theories like linear arithmetic, bit-vectors, arrays, and quantifiers. For
  instance, an SMT-solver can determine if a formula is satisfiable under
  conditions where all variables are integers that meet specific linear
  constraints. Similar to the Lazy Clause Generation utilized by CP-SAT,
  SMT-solvers usually use a SAT-solver in the backend, extended by complex
  encodings and additional propagators to handle an extensive portfolio of
  expressions. These solvers are commonly used in automated theorem proving and
  system verification. A popular SMT-solver is:
  - [Z3](https://github.com/z3prover/z3): Developed by Microsoft and available
    under the MIT license, Z3 offers a robust Python interface and comprehensive
    documentation, making it accessible for a variety of applications.
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
- **Modeling Languages:** Modeling languages provide a high-level, user-friendly
  interface for formulating optimization problems, focusing on the challenges of
  developing and maintaining models that accurately reflect real-world
  scenarios. These languages are solver-agnostic, allowing easy switching
  between different solvers - such as from the free SCIP solver to the
  commercial Gurobi - without modifying the underlying model. They also
  facilitate the use of diverse techniques, like transitioning between
  constraint programming and mixed integer programming. However, the trade-off
  is a potential loss of control and performance for the sake of generality and
  simplicity. Some of the most popular modeling languages include:
  - [MiniZinc](https://www.minizinc.org/): Very well-documented and free
    modelling language that seems to have a high reputation especially in the
    academic community. The
    [amazing course on constraint programming by Pierre Flener](https://user.it.uu.se/~pierref/courses/COCP/slides/)
    also uses MiniZinc. It supports many backends and there are the
    [MiniZinc Challenges](https://www.minizinc.org/challenge/), where CP-SAT won
    quite a lot of medals.
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
- **Approximation Algorithms:** For many difficult optimization problems, you
  can find scientific papers that describe approximation algorithms for them.
  These algorithms come with some guarantees of not being too far off from the
  optimal solution. Some are even proven to achieve the best possible
  guarantees. However, you should not directly try to implement such a paper
  even if it perfectly fits your problem. There are some approximation
  algorithms that are actually practical, but many are not. The guarantees are
  usually focussed on artificial worst-case scenarios, and even if the
  algorithms can be implemented, they may be beaten by simple heuristics.
  Approximation algorithms and their analysis can be quite useful for
  understanding the structure of your problem, but their direct practical use is
  limited.
- **Meta-Heuristics:** Instead of using a generic solver like CP-SAT, you can
  also try to build a custom algorithm for your problem based on some
  meta-heuristic pattern, like simulated annealing, genetic algorithms, or tabu
  search. Meta-heuristics require some coding, but once you know the pattern,
  they are quite straightforward to implement. While there are some libraries to
  generalize parts of the algorithms, you could also just write the whole
  algorithm yourself. This gives the advantage of truly understanding what is
  going on in the algorithm, but you miss a lot of advanced techniques contained
  in solvers like CP-SAT. For many optimization problems, you will have
  difficulties competing against techniques utilizing advanced solvers in terms
  of solution quality. If you just want a quick solution, meta-heuristics can be
  a good start.

As evident, there exists a diverse array of tools and techniques for solving
optimization problems. CP-SAT stands out as a versatile approach, particularly
well-suited for combinatorial optimization challenges. If you frequently
encounter these types of problems, becoming proficient with CP-SAT is highly
recommended. Its effectiveness across a broad spectrum of scenarios - excelling
in many - is remarkable for a tool that is both free and open-source.

---
