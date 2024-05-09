<!-- EDIT THIS PART VIA 07_under_the_hood.md -->

<a name="07-under-the-hood"></a>

## How does it work?

![Cover Image How](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_7.webp)

CP-SAT is a versatile _portfolio_ solver, centered around a _Lazy Clause
Generation (LCG)_ based Constraint Programming Solver, although it encompasses a
broader spectrum of technologies.

In its role as a portfolio solver, CP-SAT concurrently executes a multitude of
diverse algorithms and strategies, each possessing unique strengths and
weaknesses. These elements operate largely independently but engage in
information exchange, sharing progress when better solutions emerge or tighter
bounds become available.

While this may initially appear as an inefficient approach due to potential
redundancy, it proves highly effective in practice. The rationale behind this
lies in the inherent challenge of predicting which algorithm is best suited to
solve a given problem (No Free Lunch Theorem). Thus, the pragmatic strategy
involves running various approaches in parallel, with the hope that one will
effectively address the problem at hand. Note that you can also specify which
algorithms should be used if you already know which strategies are promising or
futile.

In contrast, Branch and Cut-based Mixed Integer Programming solvers like Gurobi
implement a more efficient partitioning of the search space to reduce
redundancy. However, they specialize in a particular strategy, which may not
always be the optimal choice, although it frequently proves to be so.

CP-SAT employs Branch and Cut techniques, including linear relaxations and
cutting planes, as part of its toolkit. Models that can be efficiently addressed
by a Mixed Integer Programming (MIP) solver are typically a good match for
CP-SAT as well. Nevertheless, CP-SAT's central focus is the implementation of
Lazy Clause Generation, harnessing SAT-solvers rather than relying primarily on
linear relaxations. As a result, CP-SAT may exhibit somewhat reduced performance
when confronted with MIP problems compared to dedicated MIP solvers. However, it
gains a distinct advantage when dealing with problems laden with intricate
logical constraints.

The concept behind Lazy Clause Generation involves the (incremental)
transformation of the problem into a SAT-formula, subsequently employing a
SAT-solver to seek a solution (or prove bounds by infeasibility). To mitigate
the impracticality of a straightforward conversion, Lazy Clause Generation
leverages an abundance of lazy variables and clauses.

Notably, the
[Cook-Levin Theorem](https://en.wikipedia.org/wiki/Cook%E2%80%93Levin_theorem)
attests that any problem within the realm of NP can be translated into a
SAT-formula. Optimization, in theory, could be achieved through a simple binary
search. However, this approach, while theoretically sound, lacks efficiency.
CP-SAT employs a more refined encoding scheme to tackle optimization problems
more effectively.

If you want to understand the inner workings of CP-SAT, you can follow the
following learning path:

1. Learn how to get a feasible solution based on boolean logics with
   SAT-solvers: Backtracking, DPLL, CDCL, VSIDS, ...
   - [Historical Overview by Armin Biere](https://youtu.be/DU44Y9Pt504) (video)
   - [Donald Knuth - The Art of Computer Programming, Volume 4, Fascicle 6: Satisfiability](https://www-cs-faculty.stanford.edu/~knuth/taocp.html)
     (book)
   - [Carsten Sinz and Tomas Baylo - Practical SAT Solving](https://baldur.iti.kit.edu/sat/#about)
     (slides)
2. Learn how to get provably optimal solutions via classical Mixed Integer
   Programming:
   - Linear Programming: Simplex, Duality, Dual Simplex, ...
     - [Understanding and Using Linear Programming](https://link.springer.com/book/10.1007/978-3-540-30717-4)
       (book)
     - [Optimization in Operations Research by Ronald Rardin](https://www.pearson.com/en-us/subject-catalog/p/optimization-in-operations-research/P200000003508/9780137982066)
       (very long book also containing Mixed Integer Programming, Heuristics,
       and advanced topics. For those who want to dive deep.)
     - [Video Series by Gurobi](https://www.youtube.com/playlist?list=PLHiHZENG6W8BeAfJfZ3myo5dsSQjEV5pJ)
   - Mixed Integer Programming: Branch and Bound, Cutting Planes, Branch and
     Cut, ...
     - [Discrete Optimization on Coursera with Pascal Van Hentenryck and Carleton Coffrin](https://www.coursera.org/learn/discrete-optimization)
       (video course, including also Constraint Programming and Heuristics)
     - [Gurobi Resources](https://www.gurobi.com/resource/mip-basics/) (website)
3. Learn the additional concepts of LCG Constraint Programming: Propagation,
   Lazy Clause Generation, ...
   - [Combinatorial Optimisation and Constraint Programming by Prof. Pierre Flener at Uppsala University in Sweden](https://user.it.uu.se/~pierref/courses/COCP/slides/)
     (slides)
   - [Talk by Peter Stuckey](https://www.youtube.com/watch?v=lxiCHRFNgno)
     (video)
   - [Paper on Lazy Clause Generation](https://people.eng.unimelb.edu.au/pstuckey/papers/cp09-lc.pdf)
     (paper)
4. Learn the details of CP-SAT:
   - [The proto-file of the parameters](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto)
     (source)
   - [The complete source code](https://github.com/google/or-tools/tree/stable/ortools/sat)
     (source)
   - [A recent talk by the developers of CP-SAT](https://www.youtube.com/live/vvUxusrUcpU?si=qVsXMq0xSOsfghTM)
     (video)
   - [A talk by the developers of CP-SAT](https://youtu.be/lmy1ddn4cyw) (video)

If you already have a background in Mixed Integer Programming, you may directly
jump into the slides of
[Combinatorial Optimisation and Constraint Programming](https://user.it.uu.se/~pierref/courses/COCP/slides/).
This is a full and detailed course on constraint programming, and will probably
take you some time to work through. However, it gives you all the knowledge you
need to understand the constraint programming part of CP-SAT.

> Originally, I wrote a short introduction into each of the topics, but I
> decided to remove them as the material I linked to is much better than what I
> could have written. You can find a backup of the old version
> [here](https://github.com/d-krupke/cpsat-primer/blob/main/old_how_does_it_work.md).

### What happens in CP-SAT on solve?

What actually happens when you execute `solver.Solve(model)`?

1. The model is read.
2. The model is verified.
3. Preprocessing (multiple iterations):
   1. Presolve (domain reduction) - Check
      [this video for SAT preprocessing](https://www.youtube.com/watch?v=ez9ArInp8w4),
      [this video for MaxSAT preprocessing](https://www.youtube.com/watch?v=xLg4hbM8ooM),
      and
      [this paper for MIP presolving](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/6037).
   2. Expanding higher-level constraints to lower-level constraints. See also
      the analogous
      [FlatZinc and Flattening](https://www.minizinc.org/doc-2.5.5/en/flattening.html).
   3. Detection of equivalent variables and
      [affine relations](https://personal.math.ubc.ca/~cass/courses/m309-03a/a1/olafson/affine_fuctions.htm).
   4. Substitute these by canonical representations
   5. Probe some variables to detect if they are actually fixed or detect
      further equivalences.
4. Load the preprocessed model into the underlying solver and create the linear
   relaxations.
5. **Search for solutions and bounds with the different solvers until the lower
   and upper bound match or another termination criterion is reached (e.g., time
   limit)**
6. Transform solution back to original model.

This is taken from [this talk](https://youtu.be/lmy1ddn4cyw?t=434) and slightly
extended.

### The use of linear programming techniques

As already mentioned before, CP-SAT also utilizes the (dual) simplex algorithm
and linear relaxations. The linear relaxation is implemented as a propagator and
potentially executed at every node in the search tree but only at lowest
priority. A significant difference to the application of linear relaxations in
branch and bound algorithms is that only some pivot iterations are performed (to
make it faster). However, as there are likely much deeper search trees and the
warm-starts are utilized, the optimal linear relaxation may still be computed,
just deeper down the tree (note that for SAT-solving, the search tree is usually
traversed DFS). At root level, even cutting planes such as Gomory-Cuts are
applied to improve the linear relaxation.

The linear relaxation is used for detecting infeasibility (IPs can actually be
more powerful than simple SAT, at least in theory), finding better bounds for
the objective and variables, and also for making branching decisions (using the
linear relaxation's objective and the reduced costs).

The used Relaxation Induced Neighborhood Search RINS (LNS worker), a very
successful heuristic, of course also uses linear programming.

### Limitations of CP-SAT

While CP-SAT is undeniably a potent solver, it does possess certain limitations
when juxtaposed with alternative techniques:

1. While proficient, it may not match the speed of a dedicated SAT-solver when
   tasked with solving SAT-formulas, although its performance remains quite
   commendable.
2. Similarly, for classical MIP-problems, CP-SAT may not outpace dedicated
   MIP-solvers in terms of speed, although it still delivers respectable
   performance.
3. Unlike MIP/LP-solvers, CP-SAT lacks support for continuous variables, and the
   workarounds to incorporate them may not always be highly efficient. In cases
   where your problem predominantly features continuous variables and linear
   constraints, opting for an LP-solver is likely to yield significantly
   improved performance.
4. CP-SAT does not offer support for lazy constraints or iterative model
   building, a feature available in MIP/LP-solvers and select SAT-solvers.
   Consequently, the application of exponential-sized models, which are common
   and pivotal in Mixed Integer Programming, may be restricted.
5. CP-SAT is limited to the Simplex algorithm and does not feature interior
   point methods. This limitation prevents it from employing polynomial time
   algorithms for certain classes of quadratic constraints, such as Second Order
   Cone constraints. In contrast, solvers like Gurobi utilize the Barrier
   algorithm to efficiently tackle these constraints in polynomial time.

CP-SAT might also exhibit inefficiency when confronted with certain constraints,
such as modulo constraints. However, it's noteworthy that I am not aware of any
alternative solver capable of efficiently addressing these specific constraints.
At times, NP-hard problems inherently pose formidable challenges, leaving us
with no alternative but to seek more manageable modeling approaches instead of
looking for better solvers.
