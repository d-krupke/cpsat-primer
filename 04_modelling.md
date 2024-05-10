<!--EDIT THIS PART VIA 04_modelling.md -->

<a name="04-modelling"></a>

## Modelling

![Cover Image Modelling](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_2.webp)

CP-SAT provides us with much more modelling options than the classical
MIP-solver. Instead of just the classical linear constraints (<=, ==, >=), we
have various advanced constraints such as `AllDifferent` or
`AddMultiplicationEquality`. This spares you the burden of modelling the logic
only with linear constraints, but also makes the interface more extensive.
Additionally, you have to be aware that not all constraints are equally
efficient. The most efficient constraints are linear or boolean constraints.
Constraints such as `AddMultiplicationEquality` can be significantly(!!!) more
expensive.

> **If you are coming from the MIP-world, you should not overgeneralize your
> experience** to CP-SAT as the underlying techniques are different. It does not
> rely on the linear relaxation as much as MIP-solvers do. Thus, you can often
> use modelling techniques that are not efficient in MIP-solvers, but perform
> reasonably well in CP-SAT. For example, I had a model that required multiple
> absolute values and performed significantly better in CP-SAT than in Gurobi
> (despite a manual implementation with relatively tight big-M values).

This primer does not have the space to teach about building good models. In the
following, we will primarily look onto a selection of useful constraints. If you
want to learn how to build models, you could take a look into the book
[Model Building in Mathematical Programming by H. Paul Williams](https://www.wiley.com/en-us/Model+Building+in+Mathematical+Programming%2C+5th+Edition-p-9781118443330)
which covers much more than you probably need, including some actual
applications. This book is of course not for CP-SAT, but the general techniques
and ideas carry over. However, it can also suffice to simply look on some other
models and try some things out. If you are completely new to this area, you may
want to check out modelling for the MIP-solver Gurobi in this
[video course](https://www.youtube.com/playlist?list=PLHiHZENG6W8CezJLx_cw9mNqpmviq3lO9).
Remember that many things are similar to CP-SAT, but not everything (as already
mentioned, CP-SAT is especially interesting for the cases where a MIP-solver
fails).

The following part does not cover all constraints. You can get a complete
overview by looking into the
[official documentation](https://developers.google.com/optimization/reference/python/sat/python/cp_model#cp_model.CpModel).
Simply go to `CpModel` and check out the `AddXXX` and `NewXXX` methods.

Resources on mathematical modelling (not CP-SAT specific):

- [Math Programming Modeling Basics by Gurobi](https://www.gurobi.com/resources/math-programming-modeling-basics/):
  Get the absolute basics.
- [Modeling with Gurobi Python](https://www.youtube.com/playlist?list=PLHiHZENG6W8CezJLx_cw9mNqpmviq3lO9):
  A video course on modelling with Gurobi. The concepts carry over to CP-SAT.
- [Model Building in Mathematical Programming by H. Paul Williams](https://www.wiley.com/en-us/Model+Building+in+Mathematical+Programming%2C+5th+Edition-p-9781118443330):
  A complete book on mathematical modelling.

> :warning: CP-SAT 9.9 recently changed its API to be more consistent with the
> commonly used Python style. Instead of `NewIntVar`, you can now also use
> `new_int_var`. This primer still uses the old style, but will be updated in
> the future. I observed cases where certain methods were not available in one
> or the other style, so you may need to switch between them for some versions.

---

**Elements:**

- [Variables](#04-modelling-variables)
  - [Domain Variables](#04-modelling-domain-variables)
- [Objectives](#04-modelling-objectives)
- [Linear Constraints](#04-modelling-linear-constraints)
- [Logical Constraints (Propositional Logic)](#04-modelling-logic-constraints)
- [Conditional Constraints (Channeling, Reification)](#04-modelling-conditional-constraints)
- [AllDifferent](#04-modelling-alldifferent)
- [Absolute Values and Max/Min](#04-modelling-absmaxmin)
- [Multiplication, Division, and Modulo](#04-modelling-multdivmod)
- [Circuit/Tour-Constraints](#04-modelling-circuit)
- [Interval/Packing/Scheduling Constraints](#04-modelling-intervals)
- [Piecewise Linear Constraints](#04-modelling-pwl)

---

<a name="04-modelling-variables"></a>

### Variables

There are two important types of variables in CP-SAT: Booleans and Integers
(which are actually converted to Booleans, but more on this later). There are
also, e.g.,
[interval variables](https://developers.google.com/optimization/reference/python/sat/python/cp_model#intervalvar),
but they are not as important and can be modelled easily with integer variables.
For the integer variables, you have to specify a lower and an upper bound.

```python
# integer variable z with bounds -100 <= z <= 100
z = model.NewIntVar(-100, 100, "z")
# boolean variable b
b = model.NewBoolVar("b")
# implicitly available negation of b:
not_b = b.Not()  # will be 1 if b is 0 and 0 if b is 1
```

> Having tight bounds on the integer variables can make a huge impact on the
> performance. It may be useful to run some optimization heuristics beforehand
> to get some bounds. Reducing it by a few percent can already pay off for some
> problems.

There are no continuous/floating point variables (or even constants) in CP-SAT:
If you need floating point numbers, you have to approximate them with integers
by some resolution. For example, you could simply multiply all values by 100 for
a step size of 0.01. A value of 2.35 would then be represented by 235. This
_could_ probably be implemented in CP-SAT directly, but doing it explicitly is
not difficult, and it has numerical implications that you should be aware of.

The lack of continuous variables may sound like a serious limitation, especially
if you have a background in linear optimization (where continuous variables are
the "easy part"), but as long as they are not a huge part of your problem, you
can often work around it. I had problems with many continuous variables on which
I had to apply absolute values and conditional linear constraints, and CP-SAT
performed much better than Gurobi, which is known to be very good at continuous
variables. In this case, CP-SAT struggled less with the continuous variables
(Gurobi's strength), than Gurobi with the logical constraints (CP-SAT's
strength). In a further analysis, I noted an only logarithmic increase of the
runtime with the resolution. However, there are also problems for which a higher
resolution can drastically increase the runtime. The packing problem, which is
discussed further below, has the following runtime for different resolutions:
1x: 0.02s, 10x: 0.7s, 100x: 7.6s, 1000x: 75s, 10_000x: >15min. The solution was
always the same, just scaled, and there was no objective, i.e., only a feasible
solution had to be found. Note that this is just an example, not a
representative benchmark. See
[./examples/add_no_overlap_2d_scaling.ipynb](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_no_overlap_2d_scaling.ipynb)
for the code. If you have a problem with a lot of continuous variables, such as
[network flow problems](https://en.wikipedia.org/wiki/Network_flow_problem), you
are probably better served with a MIP-solver.

In my experience, boolean variables are by far the most important variables in
many combinatorial optimization problems. Many problems, such as the famous
Traveling Salesman Problem, only consist of boolean variables. Implementing a
solver specialized on boolean variables by using a SAT-solver as a base, such as
CP-SAT, thus, is quite sensible. The resolution of coefficients (in combination
with boolean variables) is less critical than for variables.

You might question the need for naming variables in your model. While it is true
that CP-SAT would not need named variables to work (as it could just give them
automatically generated names), assigning names is incredibly useful for
debugging purposes. Solver APIs often create an internal representation of your
model, which is subsequently used by the solver. There are instances where you
might need to examine this internal model, such as when debugging issues like
infeasibility. In such scenarios, having named variables can significantly
enhance the clarity of the internal representation, making your debugging
process much more manageable.

<a name="04-modelling-domain-variables"></a>

#### Domain Variables

When dealing with integer variables that you know will only need to take certain
values, or when you wish to limit their possible values, domain variables can
become interesting. Unlike regular integer variables, domain variables are
tailored to represent a specific set of values. This approach can enhance
efficiency when the domain - the range of sensible values - is small. However,
it may not be the best choice for larger domains.

CP-SAT works by converting all integer variables into boolean variables
(warning: simplification). For each potential value, it creates two boolean
variables: one indicating whether the integer variable is equal to this value,
and another indicating whether it is less than or equal to it. This is called an
_order encoding_. At first glance, this might suggest that using domain
variables is always preferable, as it appears to reduce the number of boolean
variables needed.

However, CP-SAT employs a lazy creation strategy for these boolean variables.
This means it only generates them as needed, based on the solver's
decision-making process. Therefore, an integer variable with a wide range - say,
from 0 to 100 - will not immediately result in 200 boolean variables. It might
lead to the creation of only a few, depending on the solver's requirements.

Limiting the domain of a variable can have drawbacks. Firstly, defining a domain
explicitly can be computationally costly and increase the model size drastically
as it now need to contain not just a lower and upper bound for a variable but an
explicit list of numbers (model size is often a limiting factor). Secondly, by
narrowing down the solution space, you might inadvertently make it more
challenging for the solver to find a viable solution. First, try to let CP-SAT
handle the domain of your variables itself and only intervene if you have a good
reason to do so.

If you choose to utilize domain variables for their benefits in specific
scenarios, here is how to define them:

```python
from ortools.sat.python import cp_model

# Define a domain with selected values
domain = cp_model.Domain.FromValues([2, 5, 8, 10, 20, 50, 90])
# cam also be done via intervals
domain_2 = cp_model.Domain.FromIntervals([(8, 12), (14, 20)])

# Create a domain variable within this defined domain
x = model.NewIntVarFromDomain(domain, "x")
```

This example illustrates the process of creating a domain variable `x` that can
only take on the values specified in `domain`. This method is particularly
useful when you are working with variables that only have a meaningful range of
possible values within your problem's context.

<a name="04-modelling-objectives"></a>

### Objectives

Not every problem actually has an objective, sometimes you only need to find a
feasible solution. CP-SAT is pretty good at doing that (MIP-solvers are often
not). However, CP-SAT can also optimize pretty well (older constraint
programming solver cannot, at least in my experience). You can minimize or
maximize a linear expression (use auxiliary variables and constraints to model
more complicated expressions).

You can specify the objective function by calling `model.Minimize` or
`model.Maximize` with a linear expression.

```python
model.Maximize(30 * x + 50 * y)
```

Let us look on how to model more complicated expressions, using boolean
variables and generators.

```python
x_vars = [model.NewBoolVar(f"x{i}") for i in range(10)]
model.Minimize(
    sum(i * x_vars[i] if i % 2 == 0 else i * x_vars[i].Not() for i in range(10))
)
```

This objective evaluates to

```math
\min \sum_{i=0}^{9} i\cdot x_i \text{ if } i \text{ is even else } i\cdot \neg x_i
```

To implement a
[lexicographic optimization](https://en.wikipedia.org/wiki/Lexicographic_optimization),
you can do multiple rounds and always fix the previous objective as constraint.

```python
model.Maximize(30 * x + 50 * y)

# Lexicographic
solver.Solve(model)
model.Add(30 * x + 50 * y == int(solver.ObjectiveValue()))  # fix previous objective
model.Minimize(z)  # optimize for second objective
solver.Solve(model)
```

To implement non-linear objectives, you can use auxiliary variables and
constraints. For example, you can create a variable that is the absolute value
of another variable and then use this variable in the objective.

```python
abs_x = model.NewIntVar(0, 100, "|x|")
model.AddAbsEquality(target=abs_x, expr=x)
model.Minimize(abs_x)
```

The available constraints are discussed next.

<a name="04-modelling-linear-constraints"></a>

### Linear Constraints

These are the classical constraints also used in linear optimization. Remember
that you are still not allowed to use floating point numbers within it. Same as
for linear optimization: You are not allowed to multiply a variable with
anything else than a constant and also not to apply any further mathematical
operations.

```python
model.Add(10 * x + 15 * y <= 10)
model.Add(x + z == 2 * y)

# This one actually is not linear but still works.
model.Add(x + y != z)

# For <, > you can simply use <= and -1 because we are working on integers.
model.Add(x <= z - 1)  # x < z
```

Note that `!=` can be expected slower than the other (`<=`, `>=`, `==`)
constraints, because it is not a linear constraint. If you have a set of
mutually `!=` variables, it is better to use `AllDifferent` (see below) than to
use the explicit `!=` constraints.

> :warning: If you use intersecting linear constraints, you may get problems
> because the intersection point needs to be integral. There is no such thing as
> a feasibility tolerance as in Mixed Integer Programming-solvers, where small
> deviations are allowed. The feasibility tolerance in MIP-solvers allows, e.g.,
> 0.763445 == 0.763439 to still be considered equal to counter numerical issues
> of floating point arithmetic. In CP-SAT, you have to make sure that values can
> match exactly.

<a name="04-modelling-logic-constraints"></a>

### Logical Constraints (Propositional Logic)

You can actually model logical constraints also as linear constraints, but it
may be advantageous to show your intent:

```python
b1 = model.NewBoolVar("b1")
b2 = model.NewBoolVar("b2")
b3 = model.NewBoolVar("b3")

model.AddBoolOr(b1, b2, b3)  # b1 or b2 or b3 (at least one)
model.AddBoolAnd(b1, b2.Not(), b3.Not())  # b1 and not b2 and not b3 (all)
model.AddBoolAnd(b1, ~b2, ~b3)  # Alternative notation for `Not()`
model.AddBoolXOr(b1, b2, b3)  # b1 xor b2 xor b3
model.AddImplication(b1, b2)  # b1 -> b2
```

In this context you could also mention `AddAtLeastOne`, `AddAtMostOne`, and
`AddExactlyOne`, but these can also be modelled as linear constraints.

<a name="04-modelling-conditional-constraints"></a>

### Conditional Constraints (Channeling, Reification)

Linear constraints (Add), BoolOr, and BoolAnd support being activated by a
condition. This is not only a very helpful constraint for many applications, but
it is also a constraint that is highly inefficient to model with linear
optimization ([Big M Method](https://en.wikipedia.org/wiki/Big_M_method)). My
current experience shows that CP-SAT can work much more efficiently with this
kind of constraint. Note that you only can use a boolean variable and not
directly add an expression, i.e., maybe you need to create an auxiliary
variable.

```python
model.Add(x + z == 2 * y).OnlyEnforceIf(b1)
model.Add(x + z == 10).OnlyEnforceIf([b2, b3.Not()])  # only enforce if b2 AND NOT b3

# Restrict domain of linear expression on condition
x_b = model.NewBoolVar("x_b")
x_i = model.NEwIntVar(0, 100, "x_i")
domain = model.Domain.FromValues([0, 10, 20, 30, 40, 50])
model.AddLinearExpressionInDomain(x_i, domain).OnlyEnforceIf(x_b)
```

<a name="04-modelling-alldifferent"></a>

### AllDifferent

A constraint that is often seen in Constraint Programming, but I myself was
always able to deal without it. Still, you may find it important. It forces all
(integer) variables to have a different value.

`AllDifferent` is actually the only constraint that may use a domain based
propagator (if it is not a permutation)
[[source](https://youtu.be/lmy1ddn4cyw?t=624)]

```python
model.AddAllDifferent(x, y, z)

# You can also add a constant to the variables.
vars = [model.NewIntVar(0, 10) for i in range(10)]
model.AddAllDifferent(x + i for i, x in enumerate(vars))
```

The [N-queens](https://developers.google.com/optimization/cp/queens) example of
the official tutorial makes use of this constraint.

There is a big caveat with this constraint: CP-SAT now has a preprocessing step
that automatically tries to infer large `AllDifferent` constraints from sets of
mutual `!=` constraints. This inference equals the NP-hard Edge Clique Cover
problem, thus, is not a trivial task. If you add an `AllDifferent` constraint
yourself, CP-SAT will assume that you already took care of this inference and
will skip this step. Thus, adding a single `AllDifferent` constraint can make
your model significantly slower, if you also use `!=` constraints. If you do not
use `!=` constraints, you can safely use `AllDifferent` without any performance
penalty. You may also want to use `!=` instead of `AllDifferent` if you apply it
to overlapping sets of variables without proper optimization, because then
CP-SAT will do the inference for you.

In
[./examples/add_all_different.ipynb](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_all_different.ipynb)
you can find a quick experiment based on the graph coloring problem. In the
graph coloring problem, the colors of two adjacent vertices have to be
different. This can be easily modelled by `!=` or `AllDifferent` constraints on
every edge. Using `!=`, we can solve the example graph in around 5 seconds. If
we use `AllDifferent`, it takes more than 5 minutes. If we manually disable the
`AllDifferent` inference, it also takes more than 5 minutes. Same if we add just
a single `AllDifferent` constraint. Thus, if you use `AllDifferent` do it
properly on large sets, or use `!=` constraints and let CP-SAT infer the
`AllDifferent` constraints for you.

Maybe CP-SAT will allow you to use `AllDifferent` without any performance
penalty in the future, but for now, you have to be aware of this. See also
[the optimization parameter documentation](https://github.com/google/or-tools/blob/1d696f9108a0ebfd99feb73b9211e2f5a6b0812b/ortools/sat/sat_parameters.proto#L542).

<a name="04-modelling-absmaxmin"></a>

### Absolute Values and Max/Min

Two often occurring and important operators are absolute values as well as
minimum and maximum values. You cannot use operators directly in the
constraints, but you can use them via an auxiliary variable and a dedicated
constraint. These constraints are more efficient than comparable constraints in
classical MIP-solvers, but you should still not overuse them.

```python
# abs_xz == |x+z|
abs_xz = model.NewIntVar(0, 200, "|x+z|")  # ub = ub(x)+ub(z)
model.AddAbsEquality(target=abs_xz, expr=x + z)
# max_xyz = max(x,y,z)
max_xyz = model.NewIntVar(0, 100, "max(x,y, z)")
model.AddMaxEquality(max_xyz, [x, y, z])
# min_xyz = min(x,y,z)
min_xyz = model.NewIntVar(-100, 100, " min(x,y, z)")
model.AddMinEquality(min_xyz, [x, y, z])
```

Also note that surprisingly often, you can replace these constraints with more
efficient linear constraints. Here is one example for the max equality:

```python
x = model.NewIntVar(0, 100, "x")
y = model.NewIntVar(0, 100, "y")
z = model.NewIntVar(0, 100, "z")

# enforce that max_xyz has to be at least the maximum of x, y, z
max_xyz = model.NewIntVar(0, 100, "max_xyz")
model.Add(max_xyz >= x)
model.Add(max_xyz >= y)
model.Add(max_xyz >= z)

# as we minimized max_xyz, it has to be the maximum of x, y, z
model.Minimize(max_xyz)
```

This example illustrates that enforcing the exact maximum value is not always
necessary; a lower bound suffices. By minimizing the variable, the model itself
enforces tightness. Although this approach requires more constraints, it
utilizes constraints that are significantly more efficient than the
`AddMaxEquality` constraint, typically resulting in faster solving times.

Additional techniques exist for managing minimum and absolute values, as well as
for complex scenarios where the objective function does not directly enforce
equality. Experienced optimizers can often swiftly identify opportunities to
replace standard constraints with more efficient alternatives. However,
employing these advanced techniques should follow the acquisition of sufficient
experience or the use of a verified base model for comparison. From my
consulting experience with optimization models, I have found that resolving
issues from improperly applied optimizations frequently requires more time than
applying these techniques initially to a model that uses the less efficient
constraints.

<a name="04-modelling-multdivmod"></a>

### Multiplication, Division, and Modulo

A big nono in linear optimization (the most successful optimization area) are
multiplication of variables (because this would no longer be linear, right...).
Often we can linearize the model by some tricks and tools like Gurobi are also
able to do some non-linear optimization ( in the end, it is most often
translated to a less efficient linear model again). CP-SAT can also work with
multiplication and modulo of variables, again as constraint not as operation. So
far, I have not made good experience with these constraints, i.e., the models
end up being slow to solve, and would recommend to only use them if you really
need them and cannot find a way around them.

```python
xyz = model.NewIntVar(-100 * 100 * 100, 100**3, "x*y*z")
model.AddMultiplicationEquality(xyz, [x, y, z])  # xyz = x*y*z
model.AddModuloEquality(x, y, 3)  # x = y % 3
model.AddDivisionEquality(x, y, z)  # x = y // z
```

You can very often approximate these constraints with significantly more
efficient linear constraints, even if it may require some additional variables
or reification. Doing a piecewise linear approximation can be an option even for
more complex functions, though they too are not necessarily efficient.

Certain quadratic constraints, e.g., second-order cones, can be efficiently
handled by interior point methods, as utilized by the Gurobi solver. However,
CP-SAT currently lacks this capability and needs to do significantly more work
to handle these constraints. Long story short, if you can avoid these
constraints, you should do so, even if you have to give up on modelling the
exact function you had in mind.

> :warning: The documentation indicates that multiplication of more than two
> variables is supported, but I got an error when trying it out. I have not
> investigated this further, as I would expect it to be painfully slow anyway.

<a name="04-modelling-circuit"></a>

### Circuit/Tour-Constraints

The
[Traveling Salesman Problem (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
or Hamiltonicity Problem are important and difficult problems that occur as
subproblem in many contexts. For solving the classical TSP, you should use the
extremely powerful solver
[Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html). There is also a
separate [part in OR-Tools](https://developers.google.com/optimization/routing)
dedicated to routing. If it is just a subproblem, you can add a simple
constraint by encoding the allowed edges as triples of start vertex index,
target vertex index, and literal/variable. Note that this is using directed
edges/arcs. By adding a triple (v,v,var), you can allow CP-SAT to skip the
vertex v.

> If the tour-problem is the fundamental part of your problem, you may be better
> served with using a Mixed Integer Programming solver. Do not expect to solve
> tours much larger than 250 vertices with CP-SAT.

```python
from ortools.sat.python import cp_model

# Weighted, directed graph as instance
# (source, destination) -> cost
dgraph = {
    (0, 1): 13,
    (1, 0): 17,
    (1, 2): 16,
    (2, 1): 19,
    (0, 2): 22,
    (2, 0): 14,
    (3, 0): 15,
    (3, 1): 28,
    (3, 2): 25,
    (0, 3): 24,
    (1, 3): 11,
    (2, 3): 27,
}
model = cp_model.CpModel()
# Variables: Binary decision variables for the edges
edge_vars = {(u, v): model.NewBoolVar(f"e_{u}_{v}") for (u, v) in dgraph.keys()}
# Constraints: Add Circuit constraint
# We need to tell CP-SAT which variable corresponds to which edge.
# This is done by passing a list of tuples (u,v,var) to AddCircuit.
circuit = [
    (u, v, var) for (u, v), var in edge_vars.items()  # (source, destination, variable)
]
model.AddCircuit(circuit)

# Objective: minimize the total cost of edges
obj = sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items())
model.Minimize(obj)

# Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)
assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
tour = [(u, v) for (u, v), x in edge_vars.items() if solver.Value(x)]
print("Tour:", tour)
```

    Tour: [(0, 1), (2, 0), (3, 2), (1, 3)]

You can use this constraint very flexibly for many tour problems. We added three
examples:

- [./examples/add_circuit.py](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_circuit.py):
  The example above, slightly extended. Find out how large you can make the
  graph.
- [./examples/add_circuit_budget.py](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_circuit_budget.py):
  Find the largest tour with a given budget. This will be a bit more difficult
  to solve.
- [./examples/add_circuit_multi_tour.py](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_circuit_multi_tour.py):
  Allow $k$ tours, which in sum need to be minimal and cover all vertices.

The most powerful TSP-solver _concorde_ uses a linear programming based
approach, but with a lot of additional techniques to improve the performance.
The book _In Pursuit of the Traveling Salesman_ by William Cook may have already
given you some insights. For more details, you can also read the more advanced
book _The Traveling Salesman Problem: A Computational Study_ by Applegate,
Bixby, Chvatál, and Cook. If you need to solve some variant, MIP-solvers (which
could be called a generalization of that approach) are known to perform well
using the
[Dantzig-Fulkerson-Johnson Formulation](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Dantzig%E2%80%93Fulkerson%E2%80%93Johnson_formulation).
This model is theoretically exponential, but using lazy constraints (which are
added when needed), it can be solved efficiently in practice. The
[Miller-Tucker-Zemlin formulation](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation[21])
allows a small formulation size, but is bad in practice with MIP-solvers due to
its weak linear relaxations. Because CP-SAT does not allow lazy constraints, the
Danzig-Fulkerson-Johnson formulation would require many iterations and a lot of
wasted resources. As CP-SAT does not suffer as much from weak linear relaxations
(replacing Big-M by logic constraints, such as `OnlyEnforceIf`), the
Miller-Tucker-Zemlin formulation may be an option in some cases, though a simple
experiment (see below) shows a similar performance as the iterative approach.
When using `AddCircuit`, CP-SAT will actually use the LP-technique for the
linear relaxation (so using this constraint may really help, as otherwise CP-SAT
will not know that your manual constraints are actually a tour with a nice
linear relaxation), and probably has the lazy constraints implemented
internally. Using the `AddCircuit` constraint is thus highly recommendable for
any circle or path constraints.

In
[./examples/add_circuit_comparison.ipynb](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_circuit_comparison.ipynb),
we compare the performance of some models for the TSP, to estimate the
performance of CP-SAT for the TSP.

- **AddCircuit** can solve the Euclidean TSP up to a size of around 110 vertices
  in 10 seconds to optimality.
- **MTZ (Miller-Tucker-Zemlin)** can solve the eculidean TSP up to a size of
  around 50 vertices in 10 seconds to optimality.
- **Dantzig-Fulkerson-Johnson via iterative solving** can solve the eculidean
  TSP up to a size of around 50 vertices in 10 seconds to optimality.
- **Dantzig-Fulkerson-Johnson via lazy constraints in Gurobi** can solve the
  eculidean TSP up to a size of around 225 vertices in 10 seconds to optimality.

This tells you to use a MIP-solver for problems dominated by the tour
constraint, and if you have to use CP-SAT, you should definitely use the
`AddCircuit` constraint.

> These are all naive implementations, and the benchmark is not very rigorous.
> These values are only meant to give you a rough idea of the performance.
> Additionally, this benchmark was regarding proving _optimality_. The
> performance in just optimizing a tour could be different. The numbers could
> also look different for differently generated instances. You can find a more
> detailed benchmark in the later section on proper evaluation.

Here is the performance of `AddCircuit` for the TSP on some instances (rounded
eucl. distance) from the TSPLIB with a time limit of 90 seconds.

| Instance | # nodes | runtime | lower bound | objective | opt. gap |
| :------- | ------: | ------: | ----------: | --------: | -------: |
| att48    |      48 |    0.47 |       33522 |     33522 |        0 |
| eil51    |      51 |    0.69 |         426 |       426 |        0 |
| st70     |      70 |     0.8 |         675 |       675 |        0 |
| eil76    |      76 |    2.49 |         538 |       538 |        0 |
| pr76     |      76 |   54.36 |      108159 |    108159 |        0 |
| kroD100  |     100 |    9.72 |       21294 |     21294 |        0 |
| kroC100  |     100 |    5.57 |       20749 |     20749 |        0 |
| kroB100  |     100 |     6.2 |       22141 |     22141 |        0 |
| kroE100  |     100 |    9.06 |       22049 |     22068 |        0 |
| kroA100  |     100 |    8.41 |       21282 |     21282 |        0 |
| eil101   |     101 |    2.24 |         629 |       629 |        0 |
| lin105   |     105 |    1.37 |       14379 |     14379 |        0 |
| pr107    |     107 |     1.2 |       44303 |     44303 |        0 |
| pr124    |     124 |    33.8 |       59009 |     59030 |        0 |
| pr136    |     136 |   35.98 |       96767 |     96861 |        0 |
| pr144    |     144 |   21.27 |       58534 |     58571 |        0 |
| kroB150  |     150 |   58.44 |       26130 |     26130 |        0 |
| kroA150  |     150 |   90.94 |       26498 |     26977 |       2% |
| pr152    |     152 |   15.28 |       73682 |     73682 |        0 |
| kroA200  |     200 |   90.99 |       29209 |     29459 |       1% |
| kroB200  |     200 |   31.69 |       29437 |     29437 |        0 |
| pr226    |     226 |   74.61 |       80369 |     80369 |        0 |
| gil262   |     262 |   91.58 |        2365 |      2416 |       2% |
| pr264    |     264 |   92.03 |       49121 |     49512 |       1% |
| pr299    |     299 |   92.18 |       47709 |     49217 |       3% |
| linhp318 |     318 |   92.45 |       41915 |     52032 |      19% |
| lin318   |     318 |   92.43 |       41915 |     52025 |      19% |
| pr439    |     439 |   94.22 |      105610 |    163452 |      35% |

<a name="04-modelling-array"></a>

### Array operations

You can even go completely bonkers and work with arrays in your model. The
element at a variable index can be accessed via an `AddElement` constraint.

The second constraint is actually more of a stable matching in array form. For
two arrays of variables $v,w, |v|=|w|$, it requires
$v[i]=j \Leftrightarrow w[j]=i \quad \forall i,j \in 0,\ldots,|v|-1$. Note that
this restricts the values of the variables in the arrays to $0,\ldots, |v|-1$.

```python
# ai = [x,y,z][i]  assign ai the value of the i-th entry.
ai = model.NewIntVar(-100, 100, "a[i]")
i = model.NewIntVar(0, 2, "i")
model.AddElement(index=i, variables=[x, y, z], target=ai)

model.AddInverse([x, y, z], [z, y, x])
```

<a name="04-modelling-intervals"></a>

### Interval Variables and No-Overlap Constraints

A special case of variables are the interval variables, that allow to model
intervals, i.e., a span of some length with a start and an end. There are fixed
length intervals, flexible length intervals, and optional intervals to model
various use cases. These intervals become interesting in combination with the
no-overlap constraints for 1D and 2D. We can use this for geometric packing
problems, scheduling problems, and many other problems, where we have to prevent
overlaps between intervals. These variables are special because they are
actually not a variable, but a container that bounds separately defined start,
length, and end variables.

```python
from ortools.sat.python import cp_model

start_var = model.NewIntVar(0, 100, "start")
length_var = model.NewIntVar(10, 20, "length")
end_var = model.NewIntVar(0, 100, "end")
is_present_var = model.NewBoolVar("is_present")

# creating an interval of fixed length
fixed_interval = model.NewFixedSizeIntervalVar(
    start=start_var, size=10, end=end_var, name="fixed_interval"
)
# creating an interval whose length can be influenced by a variable (more expensive)
flexible_interval = model.NewIntervalVar(
    start=start_var, size=length_var, end=end_var, name="flexible_interval"
)
# creating an interval that can be present or not
optional_fixed_interval = model.NewOptionalFixedSizeIntervalVar(
    start=start_var,
    size=10,
    end=end_var,
    is_present=is_present_var,
    name="optional_fixed_interval",
)
# creating an interval that can be present or not and whose length can be influenced by a variable (most expensive)
optional_interval = model.NewOptionalIntervalVar(
    start=start_var,
    size=length_var,
    end=end_var,
    is_present=is_present_var,
    name="optional_interval",
)
```

There are now the two no-overlap constraints for 1D and 2D that can be used to
prevent overlaps between intervals. The 1D no-overlap constraint is used to
prevent overlaps between intervals on a single dimension, e.g., time. The 2D
no-overlap constraint is used to prevent overlaps between intervals on two
dimensions, e.g., time and resources or for packing rectangles.

```python
# 1D no-overlap constraint
model.AddNoOverlap([__INTERVAL_VARS__])
# 2D no-overlap constraint. The two lists need to have the same length.
model.AddNoOverlap2D(
    [__INTERVAL_VARS_FIRST_DIMENSION__], [__INTERVAL_VARS_SECOND_DIMENSION__]
)
```

Let us take a quick look on how we can use this to check if we can pack a set of
rectangles into a container without overlaps. This can be an interesting problem
in logistics, where we have to pack boxes into a container, or in cutting stock
problems, where we have to cut pieces from a larger piece of material.

```python
class RectanglePackingWithoutRotationsModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.model = cp_model.CpModel()

        # We have to create the variable for the bottom left corner of the boxes.
        # We directly limit their range, such that the boxes are inside the container
        self.x_vars = [
            self.model.NewIntVar(
                0, instance.container.width - box.width, name=f"x1_{i}"
            )
            for i, box in enumerate(instance.rectangles)
        ]
        self.y_vars = [
            self.model.NewIntVar(
                0, instance.container.height - box.height, name=f"y1_{i}"
            )
            for i, box in enumerate(instance.rectangles)
        ]

        # Interval variables are actually more like constraint containers, that are then passed to the no overlap constraint
        # Note that we could also make size and end variables, but we do not need them here
        x_interval_vars = [
            self.model.NewFixedSizeIntervalVar(
                start=self.x_vars[i],  # the x value of the bottom left corner
                size=box.width,  # the width of the rectangle
                name=f"x_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        y_interval_vars = [
            self.model.NewFixedSizeIntervalVar(
                start=self.y_vars[i],  # the y value of the bottom left corner
                size=box.height,  # the height of the rectangle
                name=f"y_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        # Enforce that no two rectangles overlap
        self.model.AddNoOverlap2D(x_interval_vars, y_interval_vars)

    def _extract_solution(self, solver: cp_model.CpSolver) -> Optional[Solution]:
        if self.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        placements = []
        for i, box in enumerate(self.instance.rectangles):
            x = solver.Value(self.x_vars[i])
            y = solver.Value(self.y_vars[i])
            placements.append(Placement(x=x, y=y))
        return Solution(placements=placements)

    def solve(self, time_limit: float = 900.0):
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = time_limit
        self.status = solver.Solve(self.model)
        self.solution = self._extract_solution(solver)
        self.upper_bound = solver.BestObjectiveBound()
        self.objective_value = solver.ObjectiveValue()
        return self.status

    def is_infeasible(self):
        return self.status == cp_model.INFEASIBLE

    def is_feasible(self):
        return self.status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
```

The optional intervals with flexible length allow us to even model rotations and
instead of just checking if a feasible packing exists, finding the largest
possible packing. The code may look a bit more complex, but considering the
complexity of the problem, it is still quite simple.

```python
class RectangleKnapsackWithRotationsModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.model = cp_model.CpModel()

        # Create coordinates for the placement. We need variables for the begin and end of each rectangle.
        # This will also ensure that the rectangles are placed inside the container.
        self.bottom_left_x_vars = [
            self.model.NewIntVar(0, instance.container.width, name=f"x1_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        self.bottom_left_y_vars = [
            self.model.NewIntVar(0, instance.container.height, name=f"y1_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        self.upper_right_x_vars = [
            self.model.NewIntVar(0, instance.container.width, name=f"x2_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        self.upper_right_y_vars = [
            self.model.NewIntVar(0, instance.container.height, name=f"y2_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        # These variables indicate if a rectangle is rotated or not
        self.rotated_vars = [
            self.model.NewBoolVar(f"rotated_{i}")
            for i in range(len(instance.rectangles))
        ]
        # Depending on if a rectangle is rotated or not, we have to adjust the width and height variables
        self.width_vars = [
            self.model.NewIntVar(0, max(box.width, box.height), name=f"width_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        self.height_vars = [
            self.model.NewIntVar(0, max(box.width, box.height), name=f"height_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        # Here we enforce that the width and height variables are correctly set
        for i, box in enumerate(instance.rectangles):
            if box.width > box.height:
                diff = box.width - box.height
                self.model.Add(
                    self.width_vars[i] == box.width - self.rotated_vars[i] * diff
                )
                self.model.Add(
                    self.height_vars[i] == box.height + self.rotated_vars[i] * diff
                )
            else:
                diff = box.height - box.width
                self.model.Add(
                    self.width_vars[i] == box.width + self.rotated_vars[i] * diff
                )
                self.model.Add(
                    self.height_vars[i] == box.height - self.rotated_vars[i] * diff
                )
        # And finally, a variable indicating if a rectangle is packed or not
        self.packed_vars = [
            self.model.NewBoolVar(f"packed_{i}")
            for i in range(len(instance.rectangles))
        ]

        # Interval variables are actually more like constraint containers, that are then passed to the no overlap constraint
        # Note that we could also make size and end variables, but we do not need them here
        self.x_interval_vars = [
            self.model.NewOptionalIntervalVar(
                start=self.bottom_left_x_vars[i],
                size=self.width_vars[i],
                is_present=self.packed_vars[i],
                end=self.upper_right_x_vars[i],
                name=f"x_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        self.y_interval_vars = [
            self.model.NewOptionalIntervalVar(
                start=self.bottom_left_y_vars[i],
                size=self.height_vars[i],
                is_present=self.packed_vars[i],
                end=self.upper_right_y_vars[i],
                name=f"y_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        # Enforce that no two rectangles overlap
        self.model.AddNoOverlap2D(self.x_interval_vars, self.y_interval_vars)

        # maximize the number of packed rectangles
        self.model.Maximize(
            sum(
                box.value * self.packed_vars[i]
                for i, box in enumerate(instance.rectangles)
            )
        )

    def _extract_solution(self, solver: cp_model.CpSolver) -> Optional[Solution]:
        if self.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        placements = []
        for i, box in enumerate(self.instance.rectangles):
            if solver.Value(self.packed_vars[i]):
                placements.append(
                    Placement(
                        x=solver.Value(self.bottom_left_x_vars[i]),
                        y=solver.Value(self.bottom_left_y_vars[i]),
                        rotated=bool(solver.Value(self.rotated_vars[i])),
                    )
                )
            else:
                placements.append(None)
        return Solution(placements=placements)

    def solve(self, time_limit: float = 900.0, opt_tol: float = 0.01):
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.relative_gap_limit = opt_tol
        self.status = solver.Solve(self.model)
        self.solution = self._extract_solution(solver)
        self.upper_bound = solver.BestObjectiveBound()
        self.objective_value = solver.ObjectiveValue()
        return self.status
```

|                       ![./images/dense_packing.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/dense_packing.png)                       |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| This dense packing was found by CP-SAT in less than 0.3s, which is quite impressive and seems to be more efficient than a naive Gurobi implementation. |

You can find the full code here:

|                           Problem Variant                            |                                                                                Code                                                                                 |
| :------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     Deciding feasibility of packing rectangles without rotations     |    [./evaluations/packing/solver/packing_wo_rotations.py](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/packing/solver/packing_wo_rotations.py)    |
| Finding the largest possible packing of rectangles without rotations |   [./evaluations/packing/solver/knapsack_wo_rotations.py](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/packing/solver/knapsack_wo_rotations.py)   |
|      Deciding feasibility of packing rectangles with rotations       |  [./evaluations/packing/solver/packing_with_rotations.py](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/packing/solver/packing_with_rotations.py)  |
|  Finding the largest possible packing of rectangles with rotations   | [./evaluations/packing/solver/knapsack_with_rotations.py](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/packing/solver/knapsack_with_rotations.py) |

CP-SAT is good at finding a feasible packing, but incapable of proofing
infeasibility in most cases. When using the knapsack variant, it can still pack
most of the rectangles even for the larger instances.

|                           ![./images/packing_plot_solved.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/packing_plot_solved.png)                            |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| The number of solved instances for the packing problem (90s time limit). Rotations make things slightly more difficult. None of the used instances were proofed infeasible. |
|                            ![./images/packing_percentage.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/packing_percentage.png)                             |
|                                            However, CP-SAT is able to pack nearly all rectangles even for the largest instances.                                            |

#### Resolution and Parameters

In earlier versions of CP-SAT, the performance of no-overlap constraints was
greatly influenced by the resolution. This impact has evolved, yet it remains
somewhat inconsistent. In a notebook example, I explored how resolution affects
the execution time of the no-overlap constraint in versions 9.3 and 9.8 of
CP-SAT. For version 9.3, there is a noticeable increase in execution time as the
resolution grows. Conversely, in version 9.8, execution time actually reduces
when the resolution is higher, a finding supported by repeated tests. This
unexpected behavior suggests that the performance of CP-SAT regarding no-overlap
constraints has not stabilized and may continue to vary in upcoming versions.

| Resolution | Runtime (CP-SAT 9.3) | Runtime (CP-SAT 9.8) |
| ---------- | -------------------- | -------------------- |
| 1x         | 0.02s                | 0.03s                |
| 10x        | 0.7s                 | 0.02s                |
| 100x       | 7.6s                 | 1.1s                 |
| 1000x      | 75s                  | 40.3s                |
| 10_000x    | >15min               | 0.4s                 |

[This notebook](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_no_overlap_2d.ipynb)
was used to create the table above.

However, while playing around with less documented features, I noticed that the
performance for the older version can be improved drastically with the following
parameters:

```python
solver.parameters.use_energetic_reasoning_in_no_overlap_2d = True
solver.parameters.use_timetabling_in_no_overlap_2d = True
solver.parameters.use_pairwise_reasoning_in_no_overlap_2d = True
```

With the latest version of CP-SAT, I did not notice a significant difference in
performance when using these parameters.

<a name="04-modelling-pwl"></a>

### Non-Linear Constraints/Piecewise Linear Functions

In practice, you often have cost functions that are not linear. For example,
consider a production problem where you have three different items you produce.
Each item has different components, you have to buy. The cost of the components
will first decrease with the amount you buy, then at some point increase again
as your supplier will be out of stock and you have to buy from a more expensive
supplier. Additionally, you only have a certain amount of customers willing to
pay a certain price for your product. If you want to sell more, you will have to
lower the price, which will decrease your profit.

Let us assume such a function looks like $y=f(x)$ in the following figure.
Unfortunately, it is a rather complex function that we cannot directly express
in CP-SAT. However, we can approximate it with a piecewise linear function as
shown in red. Such piecewise linear approximations are very common, and some
solvers can even do them automatically, e.g., Gurobi. The resolution can be
arbitrarily high, but the more segments you have, the more complex the model
becomes. Thus, it is usually only chosen to be as high as necessary.

|                                                                                                                     ![./images/pwla.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/pwla.png)                                                                                                                      |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| We can model an arbitrary continuous function with a piecewise linear function. Here, we split the original function into a number of straight segments. The accuracy can be adapted to the requirements. The linear segments can then be expressed in CP-SAT. The fewer such segments, the easier it remains to model and solve. |

Using linear constraints (`model.Add`) and reification (`.OnlyEnforceIf`), we
can model such a piecewise linear function in CP-SAT. For this we simply use
boolean variables to decide for a segment, and then activate the corresponding
linear constraint via reification. However, this has two problems in CP-SAT, as
shown in the next figure.

|                                                                                                             ![./images/pwla_problems.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/pwla_problems.png)                                                                                                              |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Even if the function f(x) now consists of linear segments, we cannot simply implement $y=f(x)$ in CP-SAT. First, for many $x$-values, $f(x)$ will be not integral and, thus, infeasible. Second, the canonical representation of many linear segments will require non-integral coefficients, which are also not allowed in CP-SAT. |

- **Problem A:** Even if we can express a segment as a linear function, the
  result of the function may not be integral. In the example, $f(5)$ would be
  $3.5$ and, thus, if we enforce $y=f(x)$, $x$ would be prohibited to be $5$,
  which is not what we want. There are two options now. Either, we use a more
  complex piecewise linear approximation that ensures that the function will
  always yield integral solutions or we use inequalities instead. The first
  solution has the issue that this can require too many segments, making it far
  too expensive to optimize. The second solution will be a weaker constraint as
  now we can only enforce $y<=f(x)$ or $y>=f(x)$, but not $y=f(x)$. If you try
  to enforce it by $y<=f(x)$ and $y>=f(x)$, you will end with the same
  infeasibility as before. However, often an inequality will be enough. If the
  problem is to prevent $y$ from becoming too large, you use $y<=f(x)$, if the
  problem is to prevent $y$ from becoming too small, you use $y>=f(x)$. If we
  want to represent the costs by $f(x)$, we would use $y>=f(x)$ to minimize the
  costs.

- **Problem B:** The canonical representation of a linear function is $y=ax+b$.
  However, this will often require non-integral coefficients. Luckily, we can
  automatically scale them up to integral values by adding a scaling factor. The
  inequality $y=0.5x+0.5$ in the example can also be represented as $2y=x+1$. I
  will spare you the math, but it just requires a simple trick with the least
  common multiple. Of course, the required scaling factor can become large, and
  at some point lead to overflows.

An implementation could now look as follows:

```python
# We want to enforce y=f(x)
x = model.NewIntVar(0, 7, "x")
y = model.NewIntVar(0, 5, "y")

# use boolean variables to decide for a segment
segment_active = [model.NewBoolVar("segment_1"), model.NewBoolVar("segment_2")]
model.AddAtMostOne(segment_active)  # enforce one segment to be active

# Segment 1
# if 0<=x<=3, then y >= 0.5*x + 0.5
model.Add(2 * y >= x + 1).OnlyEnforceIf(segment_active[0])
model.Add(x >= 0).OnlyEnforceIf(segment_active[0])
model.Add(x <= 3).OnlyEnforceIf(segment_active[0])

# Segment 2
model.Add(_SLIGHTLY_MORE_COMPLEX_INEQUALITY_).OnlyEnforceIf(segment_active[1])
model.Add(x >= 3).OnlyEnforceIf(segment_active[1])
model.Add(x <= 7).OnlyEnforceIf(segment_active[1])

model.Minimize(y)
# if we were to maximize y, we would have used <= instead of >=
```

This can be quite tedious, but luckily, I wrote a small helper class that will
do this automatically for you. You can find it in
[./utils/piecewise_functions](https://github.com/d-krupke/cpsat-primer/blob/main/utils/piecewise_functions/).
Simply copy it into your code.

This code does some further optimizations:

1. Considering every segment as a separate case can be quite expensive and
   inefficient. Thus, it can make a serious difference if you can combine
   multiple segments into a single case. This can be achieved by detecting
   convex ranges, as the constraints of convex areas do not interfere with each
   other.
2. Adding the convex hull of the segments as a redundant constraint that does
   not depend on any `OnlyEnforceIf` can in some cases help the solver to find
   better bounds. `OnlyEnforceIf`-constraints are often not very good for the
   linear relaxation, and having the convex hull as independent constraint can
   directly limit the solution space, without having to do any branching on the
   cases.

Let us use this code to solve an instance of the problem above.

We have two products that each require three components. The first product
requires 3 of component 1, 5 of component 2, and 2 of component 3. The second
product requires 2 of component 1, 1 of component 2, and 3 of component 3. We
can buy up to 1500 of each component for the price given in the figure below. We
can produce up to 300 of each product and sell them for the price given in the
figure below.

| ![./images/production_example_cost_components.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/production_example_cost_components.png) | ![./images/production_example_selling_price.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/production_example_selling_price.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                Costs for buying components necessary for production.                                                 |                                                         Selling price for the products.                                                          |

We want to maximize the profit, i.e., the selling price minus the costs for
buying the components. We can model this as follows:

```python
requirements_1 = (3, 5, 2)
requirements_2 = (2, 1, 3)

from ortools.sat.python import cp_model

model = cp_model.CpModel()
buy_1 = model.NewIntVar(0, 1_500, "buy_1")
buy_2 = model.NewIntVar(0, 1_500, "buy_2")
buy_3 = model.NewIntVar(0, 1_500, "buy_3")

produce_1 = model.NewIntVar(0, 300, "produce_1")
produce_2 = model.NewIntVar(0, 300, "produce_2")

model.Add(produce_1 * requirements_1[0] + produce_2 * requirements_2[0] <= buy_1)
model.Add(produce_1 * requirements_1[1] + produce_2 * requirements_2[1] <= buy_2)
model.Add(produce_1 * requirements_1[2] + produce_2 * requirements_2[2] <= buy_3)

# You can find this code it ./utils!
from piecewise_functions import PiecewiseLinearFunction, PiecewiseLinearConstraint

# Define the functions for the costs
costs_1 = [(0, 0), (1000, 400), (1500, 1300)]
costs_2 = [(0, 0), (300, 300), (700, 500), (1200, 600), (1500, 1100)]
costs_3 = [(0, 0), (200, 400), (500, 700), (1000, 900), (1500, 1500)]
# PiecewiseLinearFunction is a pydantic model and can be serialized easily!
f_costs_1 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_1], ys=[y for x, y in costs_1]
)
f_costs_2 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_2], ys=[y for x, y in costs_2]
)
f_costs_3 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_3], ys=[y for x, y in costs_3]
)

# Define the functions for the gain
gain_1 = [(0, 0), (100, 800), (200, 1600), (300, 2_000)]
gain_2 = [(0, 0), (80, 1_000), (150, 1_300), (200, 1_400), (300, 1_500)]
f_gain_1 = PiecewiseLinearFunction(xs=[x for x, y in gain_1], ys=[y for x, y in gain_1])
f_gain_2 = PiecewiseLinearFunction(xs=[x for x, y in gain_2], ys=[y for x, y in gain_2])

# Create y>=f(x) constraints for the costs
x_costs_1 = PiecewiseLinearConstraint(model, buy_1, f_costs_1, upper_bound=False)
x_costs_2 = PiecewiseLinearConstraint(model, buy_2, f_costs_2, upper_bound=False)
x_costs_3 = PiecewiseLinearConstraint(model, buy_3, f_costs_3, upper_bound=False)

# Create y<=f(x) constraints for the gain
x_gain_1 = PiecewiseLinearConstraint(model, produce_1, f_gain_1, upper_bound=True)
x_gain_2 = PiecewiseLinearConstraint(model, produce_2, f_gain_2, upper_bound=True)

# Maximize the gain minus the costs
model.Maximize(x_gain_1.y + x_gain_2.y - (x_costs_1.y + x_costs_2.y + x_costs_3.y))

solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True
status = solver.Solve(model)
print(f"Buy {solver.Value(buy_1)} of component 1")
print(f"Buy {solver.Value(buy_2)} of component 2")
print(f"Buy {solver.Value(buy_3)} of component 3")
print(f"Produce {solver.Value(produce_1)} of product 1")
print(f"Produce {solver.Value(produce_2)} of product 2")
print(f"Overall gain: {solver.ObjectiveValue()}")
```

This will give you the following output:

```
Buy 930 of component 1
Buy 1200 of component 2
Buy 870 of component 3
Produce 210 of product 1
Produce 150 of product 2
Overall gain: 1120.0
```

Unfortunately, these problems quickly get very complicated to model and solve.
This is just a proof that, theoretically, you can model such problems in CP-SAT.
Practically, you can lose a lot of time and sanity with this if you are not an
expert.

### There is more

CP-SAT has even more constraints, but I think I covered the most important ones.
If you need more, you can check out the
[official documentation](https://developers.google.com/optimization/reference/python/sat/python/cp_model#cp_model.CpModel).

---
