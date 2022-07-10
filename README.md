# Using and Understanding ortools' CP-SAT: A Primer and Cheat Sheet

*By Dominik Krupke, TU Braunschweig*

**This tutorial is under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Smaller parts can be copied without
any acknowledgements for non-commerical, educational purposes. Feel free to contribute.**

> **This text is an early draft with lots of mistakes**

This tutorial shall help you use and understand the LCG-based Constraint Programming
Solver [CP-SAT of Google's ortools suite](https://github.com/google/or-tools/).
CP-SAT is a new generation of constraint programming solvers that can actually compete for some optimization problems
against classical [Branch and Bound](https://en.wikipedia.org/wiki/Branch_and_bound)
& [Cut](https://en.wikipedia.org/wiki/Branch_and_cut)-approaches, e.g., [Gurobi](https://www.gurobi.com/).
Unfortunately, the CP-SAT does not yet have the maturity of established tools such as Gurobi and thus, the educational
material is somehow lacking (it is not bad, but maybe not sufficient for such a powerful tool).
This tutorial shall help you, especially if you are coming from
the [MIP](https://en.wikipedia.org/wiki/Integer_programming)-community, to use and understand this tool as
it may proof useful in cases where Branch and Bound performs poorly.
Note that I myself am coming from the [MIP](https://en.wikipedia.org/wiki/Integer_programming)-community and am not
related to the ortools-project.
In some cases I have to do (un-)educated guesses, which I cannot always mark appropriately in favour of the readability.

**Content:**

1. [Example](#example): A short example, showing the usage of CP-SAT.
2. [Modelling](#modelling): An overview of variables, objectives, and constraints. The constraints make the most
   important part.
3. [Parameters](#parameters): How to specify CP-SATs behavior, if needed. Timelimits, hints, assumptions,
   parallelization, ...
4. [How does it work?](): After we know what we can do with CP-SAT, we look into how CP-SAT will do all
   these things.

> This is just an unofficial primer, not a full documentation. I will provide links to more material whenever
> possible.

**Target audience:** People with some background
in [integer programming](https://en.wikipedia.org/wiki/Integer_programming)
/[linear optimization](https://en.wikipedia.org/wiki/Linear_programming), who would like to know an actually viable
alternative to [Branch and Cut](https://en.wikipedia.org/wiki/Branch_and_cut). However, I tried to make it
understandable for anyone interested
in [combinatorial optimization](https://en.wikipedia.org/wiki/Combinatorial_optimization).

## Example

Before we dive into any internals, let us take a quick look on a simple application of CP-SAT. This example is so simple
that you could solve it by hand, but know that CP-SAT would (probably) be fine with you adding a thousand (maybe even
ten or hundred-thousand) variables and constraints more.
The basic idea of using CP-SAT is, analogous to MIPs, to define an optimization problem in terms of variables,
constraints, and objective function, and then let the solver find a solution for it.
For people not familiar with this deklarative approach, you can compare it SQL where you also just state what data you
want, not how to get it.
However, it is not purely deklarativ, because it can still make a huge(!) difference how you model the problem and
getting that right takes some experience and understanding of the internals.
You can still get lucky for smaller problems (lets say few hundreds to thousands variables) and obtain optimal solutions
without having an idea of what is going on.

Our first problem has no deeper meaning but just shows some basics to get you started:

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Variables
x = model.NewIntVar(0, 100, 'x')  # you always need to specify an upper bound.
y = model.NewIntVar(0, 100, 'y')
# there are also no continuous variables: You have to decide for a resolution and then work on integers.

# Constraints
model.Add(x + y <= 30)

# Objective
model.Maximize(30 * x + 50 * y)

# Solve
solver = cp_model.CpSolver()  # Contrary to Gurobi, model and solver are separated.
status = solver.Solve(model)
assert status == cp_model.OPTIMAL  # The status tells us if we were able to compute a solution.
print(f"x={solver.Value(x)},  y={solver.Value(y)}")
print("=====Stats:======")
print(solver.SolutionInfo())
print(solver.ResponseStats())
```

    x=0,  y=30
    =====Stats:======
    default_lp
    CpSolverResponse summary:
    status: OPTIMAL
    objective: 1500
    best_bound: 1500
    booleans: 1
    conflicts: 0
    branches: 1
    propagations: 0
    integer_propagations: 2
    restarts: 1
    lp_iterations: 0
    walltime: 0.00289923
    usertime: 0.00289951
    deterministic_time: 8e-08
    gap_integral: 5.11888e-07

Pretty easy, right?

Here are some further examples, if you are not yet satisfied:

* [N-queens](https://developers.google.com/optimization/cp/queens) (this one also gives you a quick introduction to
  constraint programming, but it may be misleading because CP-SAT is no classical FD-solver. This example probably has
  been modified from the previous generation, which is also explained at the end.)
* [Employee Scheduling](https://developers.google.com/optimization/scheduling/employee_scheduling)
* [Job Shop Problem](https://developers.google.com/optimization/scheduling/job_shop)
* Much more examples can be found
  in [the official repository](https://github.com/google/or-tools/tree/stable/ortools/sat/samples) for multiple
  languages (yes, CP-SAT does support more than just Python). As the Python-Examples are named in snake-case, they are
  at the end of the list.

Ok. Now that you have seen a minimal model, let us look on what options we have to model a problem. Note that an
experienced optimizer may be able to model most problems with just the elements shown above, but showing your intentions
may help CP-SAT optimize your problem better. Contrary to Mixed Integer Programming, you also do not need to finetune
any Big-Ms (a reason to model higher-level constraints yourself, because the computer is usually not as good in that as
you).



---

## Modelling

We have much more constraints available than classical MIP-solver. As CP-SAT uses a different technique, we should also
not directly assume the same perfomance.

I cannot teach you how to properly model a problem at hand, here. This is rather an overview of what is possible, but
you need to get your hands dirty and just try out.

You can get a complete overview by looking into
the [official documentation](http://google.github.io/or-tools/python/ortools/sat/python/cp_model.html). Best simply go
to CpModel and check out the `AddXXX` and `NewXXX` methods.

If you want more about modelling, I recommend the
book [Model Building in Mathematical Programming by H. Paul Williams](https://www.wiley.com/en-us/Model+Building+in+Mathematical+Programming%2C+5th+Edition-p-9781118443330)
which covers much more than you probably need, including some actual applications. This book is of course not for
CP-SAT, but the general technics and idea cary over.

### Variables

There are two important types of variables: Integers and Booleans.
Actually, there are more (
e.g., [interval variables](http://google.github.io/or-tools/python/ortools/sat/python/cp_model.html#IntervalVar)), but
these are the two important ones, I use all the time.
There are not continuous/floating point variables (or even constants): If you need floating point numbers, you have to
round by some resolution. This necessity could probably be hidden from you, but for now you have to do it yourself.
You also have to specify a lower and an upper bound for integer variables (this would be much harder to do automatically
and can actually have a significant performance impact).
This may sound like a serious limitation, but actually I have successfully handled complex problems with lots of angles
and stuff with CP-SAT. You just have to get used to transforming all values.

```python
z = model.NewIntVar(-100, 100, 'z')
b = model.NewBoolVar('b')
not_b = b.Not()
```

> Most problems I know, actually have much more boolean variables than integer variables. Having a SAT-solver as base,
> thus, is not such a bad idea.

### Objectives

Not every problem actually has an objective, sometimes you only need to find a feasible solution.
CP-SAT is pretty good at doing that (MIP-solvers are not).
However, CP-SAT can also optimize pretty well (older constraint programming solver not, at least in my experience). You
can minimize or maximize a linear expression (use constraints to model more complicated expressions). To do a
lexicographic optimization, you can do multiple rounds and always fix the previous objective as constraint.

```python
model.Maximize(30 * x + 50 * y)

# Lexicographic
solver.Solve(model)
model.Add(30 * x + 50 * y == int(solver.ObjectiveValue()))  # fix previous objective
model.Minimize(z)  # optimize for second objective
solver.Solve(model)
```

### Linear Constraints

These are the classical constraints also used in linear optimization.
Remember that you are still not allowed to use floating point numbers within it.
Same as for linear optimization: You are not allowed to multiply a variable with anything else than a constant and also
not apply any further mathematical operations.

```python
model.Add(10 * x + 15 * y <= 10)
model.Add(x + z == 2 * y)

# This one actually isn't linear but still works.
model.Add(x + y != z)

# For <, > you can simply use <= and -1 because we are working on integers.
model.Add(x <= z - 1)  # x < z
```

> **Caveat:** If you use intersecting linear constraints, you may get problems because the intersection point needs to
> be integral. There is no such thing as a feasibility tolerance as in Mixed Integer Programming-solvers.

### Logical Constaints

You can actually model logical constraints also as linear constraints, but it may be advantageous to show your intent:

```python
b1 = model.NewBoolVar('b1')
b2 = model.NewBoolVar('b2')
b3 = model.NewBoolVar('b3')

model.AddBoolOr(b1, b2, b3)  # b1 or b2 or b3 (at least one)
model.AddBoolAnd(b1, b2.Not(), b3.Not())  # b1 and not b2 and not b3 (all)
model.AddBoolXOr(b1, b2, b3)  # b1 xor b2 xor b3
model.AddImplication(b1, b2)  # b1 -> b2
```

In this context you could also mention `AddAtLeastOne`, `AddAtMostOne`, and `AddExactlyOne`, but these can also be
modelled as linear constraints.

### Conditional Constraints

Linear constraints (Add), BoolOr, and BoolAnd support to be activated by a condition.
This is not only a very helpful constraint for many applications, but it is also a constraint that is highly inefficient
to model with linear optimization ([Big M Method](https://en.wikipedia.org/wiki/Big_M_method)).
My current experience shows that CP-SAT can work much more efficient with this kind of constraint.
Note that you only can use a boolean variable and not directly add an expression, i.e., maybe you need to create an
auxiliary variable.

```python
model.Add(x + z == 2 * y).OnlyEnforceIf(b1)
model.Add(x + z == 10).OnlyEnforceIf([b2, b3])  # only enforce if b2 AND b3
```

### AllDifferent

A constraint that is often see in Constraint Programming, but I myself was always able to deal without it.
Still, you may find it important. It forces all (integer) variables to have a different value.

`AllDifferent` is actually the only constraint that may use a domain based propagator (if it is not a
permutation) [[source](https://youtu.be/lmy1ddn4cyw?t=624)]

```python
model.AddAllDifferent(x, y, z)
```

### Absolute Values and Max/Min

Two often occuring and important operators are absolute values as well as minimum and maximum values.
You cannot use operators directly in the constraints, but you can use them via an auxiliary variable and a dedicated
constraint.
These constraints are reasonably efficient in my experience.

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

### Multiplication and Modulo

A big nono in linear optimization (the most successful optimization area) are multiplication of variables (because this
would no longer be linear, right...).
Often we can linearize the model by some tricks and tools like Gurobi are also able to do some non-linear optimization (
in the end, it is most often translated to a less efficient linear model again).
CP-SAT can also work with multiplication and modulo of variables, again as constraint not as operation.
I actually have no experience in how efficient this is (I try to avoid anything non-linear by experience).

```python
xyz = model.NewIntVar(-100 * 100 * 100, 100 ** 3, 'x*y*z')
model.AddMultiplicationEquality(xyz, [x, y, z])  # xyz = x*y*z
model.AddModuloEquality(x, y, 3)  # x = y % 3
```

> TODO: I don't know, if multiplication of more than two variables is actually allowed.

### Circuit/Tour-Constraints

The [Traveling Salesman Problem (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem) or Hamiltonicity
Problem are important and difficult problems that occur as subproblem in many contexts.
For solving the classical TSP, you should use the extremely powerful
solver [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html). There is also a
separate [part in ortools](https://developers.google.com/optimization/routing) dedicated to routing.
If it is just a subproblem, you can add a simple constraint by encoding the allowed edges a triples of start vertex
index, target vertex index, and literal/variable.
Note that this is using directed edges/arcs.

**If the tour-problem is the fundamental part of your problem, you may be better served with using a Mixed Integer
Programming solver. Don't expect to solve tours much larger than 250 vertices with CP-SAT.**

```python
model.AddCircuit([(0, 1, b1), (1, 0, b1), (1, 2, b2), (2, 0, b3)])
```

MIP-solver usually use something like
the [Dantzig-Fulkerson-Johnson Formulation](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Dantzig%E2%80%93Fulkerson%E2%80%93Johnson_formulation)
that potentially require an exponential amount of constraints, but perform much better than the smaller models, due to
the fact that you can add constraints lazily when needed (you usually only need a fraction of them) and the smaller
models usually rely on some kind of Big-M method. The Big-M-based models are difficult to solve with MIP-solvers due to
their weak linear relaxations. The exponential model is no option for CP-SAT because it does not allow lazy constraints.
CP-SAT does not suffer from the weak relaxations (or at least not as much), so for example
the [Miller-Tucker-Zemlin formulation](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation[21])
would be an option (despite looking nice, it is bad in practice with MIP-solver). However, with the circuit-constraint
at hand, we can also just use it and not worry about the various options of modelling tour constraints. Internally,
CP-SAT will actually use the LP-technique for the linear relaxation (so using this constraint may really help, as
otherwise CP-SAT will not know that your manual constraints are actually a tour with a nice linear relaxation).

### Array operations

You can even go completely bonkers and work with arrays in your model.
The element at a variable index can be accessed via an `AddElement` constraint.

The second constraint is actually more of a stable matching in array form.
For two arrays of variables $v,w, |v|=|w|$, it requires $v[i]=j \Leftrightarrow w[j]=i \quad \forall i,j \in
0,\ldots,|v|-1$.
Note that this restricts the values of the variables in the arrays to $0,\ldots, |v|-1$.

```python
# ai = [x,y,z][i]  assign ai the value of the i-th entry.
ai = model.NewIntVar(-100, 100, "a[i]")
i = model.NewIntVar(0, 2, "i")
model.AddElement(index=i, variables=[x, y, z], target=ai)

model.AddInverse([x, y, z], [z, y, x])
```

### But wait, there is more: Intervals and stuff

An important part we neglect to keep this tutorial reasonably short (and because I barely need them, but they are
important in some domains) are intervals.
CP-SAT has extensive support for interval variables and corresponding constraints, even two-dimensional do-not-overlap
constraints.
Maybe I add something about those later.

---

## Parameters

The CP-SAT solver has a lot of parameters to control its behavior.
They are implemented via [Protocol Buffer](https://developers.google.com/protocol-buffers) and can be manipulated via
the `parameters`-member.
If you want to find out all options, you can check the reasonably well documented `proto`-file in
the [official repository](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto).
I will give you the most important right below.

### Timelimit and Status

If we have a huge model, CP-SAT may not be able to solve it to optimality (if the constraints are not to difficult,
there is a good chance we still get a good solution).
Of course, we don't want CP-SAT to run endlessly for hours (years, decades,...) but simply abort after a fixed time and
return us the best solution so far.
If you are now asking yourself, why you should use a tool that may run forever: There are simply no provably faster
algorithms and considering the combinatorial complexity, it is incredible that it works so well.
Those not familiar with the concepts of NP-hardness and combinatorial complexty, I recommend to read the book 'In
Pursuit of the Travaling Salesman' by William Cook.
Actually, I recommend this book to anyone into optimization: It is a beautiful and light weekend-read.

To set a timelimit (in seconds), we can simply set the following value before we run the solver:

```python
solver.parameters.max_time_in_seconds = 60  # 60s timelimit
```

We now of course have the problem, that maybe we won't have an optimal solution, or a solution at all, we can continue
on.
Thus, we need to check the status of the solver.

```python
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("We have a solution.")
else:
    print("Help?! No solution available! :( ")
```

The following status codes are possible:

* `OPTIMAL`: Perfect, we have an optimal solution.
* `FEASIBLE`: Good, we have at least a feasible solution (we may also have a bound available to check the quality
  via `solver.BestObjectiveBound()`).
* `INFEASIBLE`: There is a proof that no solution can satisfy all our constraints.
* `MODEL_INVALID`: You used CP-SAT wrongly.
* `UNKNOWN`: No solution was found, but also no infeasibility proof. Thus, we actually know nothing. Maybe there is at
  least a bound available?

If you want to print the status, you can use `solver.StatusName(status)`.

We can not only limit the runtime but also tell CP-SAT, we are satisfied with a solution within a specific, proveable
quality range.
For this, we can set the parameters `absolute_gap_limit` and `relative_gap_limit`.
The absolut limit tells CP-SAT to stop as soon as the solution is at most a specific value apart to the bound, the
relative limit is relative to the bound.
More specific, CP-SAT will stop as soon as the objective value (O) is within relative ratio
$abs(O - B) / max(1, abs(O))$ of the bound (B).
To stop as soon as we are within 5% of the optimum, we could state (TODO: Check)

```python
solver.parameters.relative_gap_limit = 0.05
```

.Now we may want to stop after we didn't make progress for some time or whatever.
In this case, we can make use of the solution callbacks.

*For those familiar with Gurobi: Unfortunately, we can only abort the solution progress and not add lazy constraints or
similiar.
For those not familiar with Gurobi or MIPs: With Mixed Integer Programming we can adapt the model during the solution
process via callbacks which allows us to solve problems with many(!) constraints by only adding them lazily.*

For adding a solution callback, we have to inherit from a base class.
The documentation of the base class and the available operations can be found in
the [documentation](https://google.github.io/or-tools/python/ortools/sat/python/cp_model.html#CpSolverSolutionCallback).

```python
class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, stuff):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.stuff = stuff  # just to show that we can save some data in the callback.

    def on_solution_callback(self):
        obj = self.ObjectiveValue()  # best solution value
        bound = self.BestObjectiveBound()  # best bound
        print(f"The current value of x is {self.Value(x)}")
        if abs(obj - bound) < 10:
            self.StopSearch()  # abort search for better solution
        # ...


solver.Solve(model, MySolutionCallback(None))
```

You can find
an [official example of using such callbacks](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/stop_after_n_solutions_sample_sat.py)
.

Beside querying the objective value of the currently best solution, the solution itself, and the best known bound, you
can also find out about internals such as `NumBooleans(self)`, `NumConflicts(self)`, `NumBranches(self)`. What those
values mean will be discussed later.

### Parallelization

CP-SAT has some basic parallelization. It can be considered a portfolio-strategy with some minimal data exchange between
the threads. The basic idea is to use different techniques and may the best fitting one win.

1. The first thread performs the default search: The optimization problem is converted into a Boolean satisfiability
   problem and solved with a Variable State Independent Decaying Sum (VSIDS) algorithm. A search heuristic introduces
   additional literals for branching when needed, by selecting an integer variable, a value and a branching direction.
   The model also gets linearized to some degree, and the corresponding LP gets (partially) solved with the (dual)
   Simplex-algorithm to support the satisfiability model.
2. The second thread uses a fixed search if a decision strategy has been specified. Otherwise, it tries to follow the
   LP-branching on the linearized model.
3. The third thread uses Pseudo-Cost branching. This is a technique from mixed integer programming, where we branch on
   the variable that had the highest influence on the objective in prior branches. Of course, this only provides useful
   values after we have already performed some branches on the variable.
4. The fourth thread is like the first thread but without linear relaxation.
5. The fifth thread does the opposite and uses the default search but with maximal linear relaxation, i.e., also
   constraints that are more expensive to linearize are linearized. This can be computationally expensive but provides
   good lower bounds for some models.
6. The sixth thread performs a core based search from the SAT- community. This approach extracts unsatisfiable cores of
   the formula and is good for finding lower bounds.
7. All further threads perform a Large Neighborhood Search (LNS) for obtaining good solutions.

Note that this information may no longer be completely accurate (if it ever was).
To set the numnber of used cores/workers, simply do:

```python
solver.parameters.num_search_workers = 8  # use 8 cores
```

If you want to use more LNS-worker, you can specify `solver.parameters.min_num_lns_workers` (default 2).
You can also specify how the different cores should be used by configuring/reordering.

```
solver.parameters.subsolvers = ["default_lp", "fixed", "less_encoding", "no_lp", "max_lp", "pseudo_costs", "reduced_costs", "quick_restart", "quick_restart_no_lp", "lb_tree_search", "probing"]
 ```

This can be interesting, e.g., if you are using CP-SAT especially because the linear relaxation is not useful. There are
even more option, but for this you can simply look into
the [documentation](https://github.com/google/or-tools/blob/49b6301e1e1e231d654d79b6032e79809868a70e/ortools/sat/sat_parameters.proto#L513)
.

### Assumptions

Quite often you want to find out what happens if you force some variables to a specific value.
Because you possibly want to do that multiple times, you do not want to copy the whole model.
CP-SAT has a nice option of adding assumptions that you can clear afterwards, without needing to copy the object to test
the next assumptions.
This is a feature also known from many SAT-solvers and CP-SAT also only allows this feature for boolean literals.
You cannot add any more complicated expressions here, but for boolean literals it seems to be pretty efficient.
By adding some auxiliary boolean variables, you can also use this technique to play around with more complicated
expressions without the need to copy the model.
If you really need to temporarily add complex constraints, you may have to copy the model using `model.CopyFrom` (maybe
you also need to copy the variables. Need to check that.).

```python
model.AddAssumptions([b1, b2.Not()])  # assume b1=True, b2=False
model.AddAssumption(b3)  # assume b3=True (single literal)
# ... solve again and analyse ...
model.ClearAssumptions()  # clear all assumptions
```

> An **assumption** is a temporary fixation of a boolean variable to true or false. It can be efficiently handled by a
> SAT-solver (and thus probably also by CP-SAT) and does not harm the learned clauses (but can reuse them).

### Hints

Maybe we already have a good intuition on how the solution will look like (this could be, because we already solved a
similar model, have a good heuristic, etc.).
In this case it may be useful, to tell CP-SAT about it so it can incorporate this knowledge into its search.
For Mixed Integer Programming Solver, this often yields a visible boost, even with mediocre heuristic solutions, for
CP-SAT I actually also encountered downgrades of the performance if the hints weren't great.

```python
model.AddHint(x, 1)  # Tell CP-SAT that x will probably be 1
model.AddHint(y, 2)  # and y probably be 2.
```

You can also
find [an official example](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/solution_hinting_sample_sat.py)
.

### Decision Strategy

We can tell CP-SAT, how to branch (or make a decision) whenever it can no longer deduce anything via propagation.
For this, we need to provide a list of the variables (order may be important for some strategies), define which variable
should be selected next (fixed variables are automatically skipped), and define which value should be probed.

We have the following options for variable selection:

* `CHOOSE_FIRST`: the first not-fixed variable in the list.
* `CHOOSE_LOWEST_MIN`: the variable that could (potentially) take the lowest value.
* `CHOOSE_HIGHEST_MAX`: the variable that could (potentially) take the highest value.
* `CHOOSE_MIN_DOMAIN_SIZE`: the variable that has the fewest feasible assignments.
* `CHOOSE_MAX_DOMAIN_SIZE`: the variables the has the most feasible assignments.

For the value/domain strategy, we have the options:

* `SELECT_MIN_VALUE`: try to assign the smallest value.
* `SELECT_MAX_VALUE`: try to assign the largest value.
* `SELECT_LOWER_HALF`: branch to the lower half.
* `SELECT_UPPER_HALF`: branch to the upper half.
* `SELECT_MEDIAN_VALUE`: try to assign the median value.

> **CAVEAT:** In the documentation there is a warning about the completeness of the domain strategy. I am not sure, if
> this is just for custom strategies or you have to be careful in general. So be warned.

```python
model.AddDecisionStrategy([x], cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

# your can force CP-SAT to follow this strategy exactly
solver.parameters.search_branching = cp_model.FIXED_SEARCH
```

---

## How does it work?

Let us now take a look on what is actually happening under the hood.
You may have already learned that CP-SAT is transforming the problem into a SAT-formula.
This is of course not just an application of
the [Cook-Levin Theorem](https://en.wikipedia.org/wiki/Cook%E2%80%93Levin_theorem) and also not just creating a boolean
variable for every possible integer assignment combined with many, many constraints.
No, it is actually kind of a simulation of branch and bound on a SAT-solver (gifting us clause learning and stuff) with
a lot (!) of lazy variables and clauses (LCG).
Additionally, tools from classical linear optimization (linear relaxations, RINS, ...) are applied when useful to guide
the process (it is not like everything is done by the SAT-solver).

Before we dig any deeper, let us first get some prerequisites, so we are on the same page.
Remember, that this tutorial is written from the view of linear optimization.

### Prerequisites

CP-SAT actually builds upon quite a set of techniques. However, it is enough if you understand the basics of those.

| **WARNING** This is still in a very drafty state. There are actually good examples to list here.

#### SAT-Solvers

Today's SAT-solvers have become quite powerful and are now able to frequently solve instances with millions of variables
and clauses.
The advent of performant SAT-solvers only came around 2000 and the improvements still have some momentum.
You can get a good overview of the history and developments of SAT-solvers in [this video](https://youtu.be/DU44Y9Pt504)
by Armin Biere.
Remember that SAT-formulas are usually stated in [CNF](https://en.wikipedia.org/wiki/Conjunctive_normal_form), i.e., a
conjunction of disjunctions of literals, e.g., 
$(x_1 \vee x_2 \vee x_3) \wedge (\overline{x_1} \vee \overline{x_2})\wedge (x_1 \vee \overline{x_3})$.
Any SAT-formula can be efficiently converted to such a representation.

If you want to actually dig deep into SAT-solvers, luckily there is literatur for you, e.g., *Donald Knuth - The Art of
Computer Programming, Volume 4, Fascicle 6: Satisfiability*.
The *Handbook of Satisfiability* may provide much more information, but is unfortunately pretty expensive.
If you want some free material, I like the slides
of [Carsten Sinz and Tomas Baylo - Practical SAT Solving](https://baldur.iti.kit.edu/sat/#about) quite a lot.

##### DPLL and Unit Propagation

The first important technique in solving SAT-formulas is
the [Davis–Putnam–Logemann–Loveland (DPLL) algorithm](https://en.wikipedia.org/wiki/DPLL_algorithm).
Modern SAT-solver are actually just this backtracking-based algorithm with extras.
The probably most important part to remember is the unit-propagation: If we have a clause $(x_1\vee x_2 \vee
\overline{x_3})$ and we have already set $x_1=0$ and $x_3=1$, we know that $x_2=1$.
The important thing about unit propagation is that there are highly-efficient data structures (e.g., 2-watched literals)
that can notify us whenever this happens.
This is actually a point I missed for quite some time, so I emphasize it especially for you so you don't have the same
struggles as me: A lot of the further design decision are actually just to trigger unit propagation as often as
possible.
You may want to check out [these slides](https://baldur.iti.kit.edu/sat/files/2019/l05.pdf).

##### Conflict-driven clause learning (CDCL)

One very important idea in SAT-solving
is [learning new clauses](https://en.wikipedia.org/wiki/Conflict-driven_clause_learning), which allows us to identify
infeasibility earlier in the search tree.
We are not learning anything that is not available in the original formular, but we learn better representations of this
information.
For example, we may learn that if $x_1=1$, $x_9$ has to be 0 which is not directly visible from the clauses.
Actually, the learned clauses will have more concrete properties that make them useful, but we don't go deeper here.
Just remember that modern SAT-solver learn from failures which allows them to cut many infeasible future branches much
earlier instead of running into the same conflicts (with just some unimportant variables changed) again and again.

Let us look on an overly simplified example:
Consider the formula
$(x_0\vee x_1)\wedge (x_2 \vee x_3)\wedge (\overline{x_0}\vee\overline{x_2})\wedge (\overline{x_1}\vee x_2\vee\overline{x_3})$.
Let us assign $x_0=0$, which results in $x_1=1$ by unit propagation.
If we now assign $x_2 = 0$, we have to assign $x_3=1$ by unit propagation, but this creates a conflict in 
$(\overline{x_1} \vee x_2 \vee \overline{x_3})$.
The core of this conflict was setting $x_0=x_2=0$, and therefore we can add the clause $(x_0 \vee x_2)$.
Actually, this specific clause is not very helpful.
In CDCL we usually extract a clause (1UIP) that will easily be triggered by the unit propagation in the next step.

For a better understanding, I recommend to take a look
at [these slides](https://baldur.iti.kit.edu/sat/files/2019/l07.pdf).

> For all the Branch and Bound-people: Clause learning can be considered as some kind of infeasibility pruning. Instead
> of having bounds that tell you, you don't have to go deeper into this branch, you have get a number of conflict
> clauses
> that tell you, that nothing feasible can come out of branches that fit any of these clauses. There
> is [some work](https://www.csc.kth.se/~jakobn/research/LearnToRelax_Constraints.pdf) in also integrating this into
> branch and cut procedures, but it is not yet used in the state-of-the-art MIP-solvers, as far as I know. CP-SAT, on
> the
> other hand, does that (plus some rudimentary branch and cutting) which maybe explains why it is so much stronger for
> some problems, especially if they have a lot of logic.

#### Linear and Integer Programming

For this topic, there is actually a [nice primer by Gurobi](https://www.gurobi.com/resource/mip-basics/).
Let me quickly recap the most important parts for CP-SAT:

* A Mixed Integer Linear Program is a subset of CP-SAT, but one that is still very powerful and can be reasonably well
  solved. It limits you to linear constraints, but you can actually convert most of the other constraints to linear
  constraints.
* A mixed integer linear program is still hard to solve, but if we allow all integral values to become fractional it
  suddenly becomes a problem that we can solve efficiently. This is called a linear relaxation, potentially further
  improved by cutting planes. The linear relaxation provides us often with very good bounds (which is why Branch and Cut
  works so well for many problems).
* Thanks to duality theory, we can even get bounds without solving the linear relaxation completely (for example if we
  just want to quickly estimate the influence of a potential branching decision).
* We can warm-start this process and slight modifications to an already solved model will only take a small extra amount
  of time to solve again.

Let us take a quick look at an example for the famous NP-hard Traveling Salesman Problem.
The idea is to solve the linear relaxation of the problem, which is a provable lower bound but maybe use edges only
fractionally (which is of course prohibited).
These fractional edges are highlighted in red in the image below.
However, this relaxation is efficiently solvable and reasonably close to a proper solution.
Next, we select a fractional edge and solve the problem once with this edge forced to one and once with this edge forced
to false.
This is called the branching step and it divides the solution space into two smaller once (cutting of all solutions
where this edge is used fractionally).
For the subproblems, we can again efficiently compute the linear relaxation.
By continuing this process on the leaf with the currently best lower bound, we end reasonably quickly by a provably
optimal solution (because all other leaves have a worse objective).
Note that for this instance with 30 points, there exists over $10^{30}$ solutions which is out of reach of any computer.
Still we managed to compute the optimal solution in just a few steps.

![tsp bnb example](./images/tsp_bnb.png)
This example has been generated with [this tool by Bill Cook](http://www.math.uwaterloo.ca/tsp/D3/bootQ.html#).
Let me again recommend the
book [In Pursuit of the Traveling Salesman by Bill Cook](https://press.princeton.edu/books/paperback/9780691163529/in-pursuit-of-the-traveling-salesman)
, which actually covers all the fundamentals of linear and integer programming you need in an easily digestible way even
suited for beginners.

> **Even if SAT is the backbone of CP-SAT, linear programming techniques are used and still play a fundamental role,
especially the linear relaxation. Also see [this talk](https://youtu.be/lmy1ddn4cyw?t=1355) by the developers.
Using `model.parameters.linearization_level` you can also specify, how much of the model should be linearized. The
importance of the LP for CP-SAT also shows in some benchmarks: Without it, only 130 problems of the MIPLIB 2017 could be
solved to optimality, with LP 244, and with portfolio parallelization even 327.**

### Lazy Clause Generation Constraint Programming

The basic idea in lazy clause generation constraint programming is to convert the problem into a (lazy) SAT-formula, and
have an additional set of propagators that dynamically add clauses to satisfy the complex constraints.

> **WARNING:** This part may be overly simplified. I have only superficial knowledge of LCG (i.e., I read documentation,
> papers, and watched some talks).

#### Encoding

Let $x$ be a variable and $D(x)$ its domain, i.e., a set of the values it can take.
In the beginning $D(x)$ will be defined by the lower and upper bound.

CP-SAT uses an order and value encoding. Thus, we have the following variables:

$$[x\leq v] \quad \forall v\in D(x)$$
$$[x=v] \quad \forall v\in D(x)$$

The inverse variables can be obtained by negation

$$[x\geq v] \equiv \neg [x\leq v-1]$$
$$[x\not=v] \equiv \neg [x=v]$$

and the following constraints that enforce consistency:

$$[x\leq v] \Rightarrow [x\leq v+1]$$
$$[x=v] \Leftrightarrow [x\leq v] \wedge [x\geq v]$$

This is linear in the size of the domain for each variable, and thus still prohibitively large.
However, we probably will only need a few values for each variables.
If only the values x=1, 7 or 20 are interesting, we could simply just create variables for those and the constraints
$[x\leq 1] \Rightarrow [x\leq 7], [x\leq 7 \Rightarrow x\leq 20], \ldots$.
When it turns out, we need more, we simply extend the model lazily.

There are a few things to talk about:

1. Why do we need the order variables $[x\leq v]$? Because otherwise we would need a quadratic amount of consistency
   constraints ($[x=v] \rightarrow [x\not=v'] \forall v\not=v' \in D(x)$).
2. Why use a unary encoding instead of a logarithmic? Because it propagates much better with unit propagation. E.g., if
   $[x\leq v]$ is set, all $[x\leq v'], v'>v$ are automatically triggered. This is much harder, if not impossible, to
   achieve if each value consist of multiple variables. Thanks to the lazy variable generation, we often still need only
   few explicit values.

#### Propagator

So, we have consistent integral variables in a SAT-formula, but how do we add numerical constraints?

Let us take a look on the simple constraint $x=y+z$.
This constraint can also be expressed as $y=x-z$ and $z=x-y$.
We can propagate the domains of the variables onto each other, especially if we fixed the value of one during search,
e.g., $D(x)={0, 1, \ldots, 100} \rightarrow D(x)=\{5\}$.

$$ x \geq \min(D(y))+\min(D(z)) \quad x \leq \max(D(y))+\max(D(z)) $$

$$ y \geq \min(D(x))-\max(D(z)) \quad y \leq \max(D(x))-\min(D(z)) $$

$$ z \geq \min(D(x))-\max(D(y)) \quad z \leq \max(D(x))-\min(D(y)) $$

In this context, the technique of [SMT Solver](https://en.wikipedia.org/wiki/Satisfiability_modulo_theories) can also be
interesting.

#### Branching/Searching

Whenever we can no longer apply any propagation, i.e., reached a fixpoint, we have to branch on some variable.
Branching is actually just fixing a variable, e.g., $[x\leq 7]=1$.
This is actually just DPLL.

For finding an optional solution, we just have to find a feasible solution with $[obj\leq T]=1$ is satisfiable and
$[obj\leq T-1]$ is unsatisfiable (for a minimization problem).

An example for LCG can be seen below.
This example is taken from a [talk of Peter Stuckey](https://youtu.be/lxiCHRFNgno?t=642) (link directly guides you to
the right position in the video, if you want this example explained to you) and shows a search process that leads to
conflict and a newly learned clause to prevent this conflict earlier in other branches.
The green literals show search decisions/branches (talking about branches is slightly odd because of the way SAT-solver
search: they usually have only a single path of the tree in memory).
The purple literals are triggered by the numeric consistency rules.
The columns with the blue headlines show the application of propagators (i.e., clause generation) for the three
constraints.
The arrows pointing towards a node can be seen as conjunctive implication clauses ($x\wedge y \Rightarrow z$).

$$x_1,x_2,x_3,x_4,x_5 \in \{0,1,2,3,4,5\}$$

$$\mathtt{AllDifferent}(x_1,x_2,x_3,x_4)$$

$$x_2\leq x_5$$

$$x_1+x_2+x_3+x_4\leq 9$$

![LCG examples](./images/lcg.png)

Note that the 1UIP is pretty great: independent of the $[[x_5\leq 2]]$ decision,
the new clause will directly trigger and set $\neg [[x_2=2]]$
(in addition to $\neg [[x_5\leq 2]]$ by search).

### What happens in CP-SAT on solve?

So, what actually happens when you execute `solver.Solve(model)`?

1. The model is read.
2. The model is verified.
3. Preprocessing (multiple iterations):
   a. Presolve (domain reduction)
   b. Expanding higher-level constraints to lower-level constraints. See also the
   analogous [FlatZinc and Flattening](https://www.minizinc.org/doc-2.5.5/en/flattening.html).
   c. Detection of equivalent variables
   and [affine relations](https://personal.math.ubc.ca/~cass/courses/m309-03a/a1/olafson/affine_fuctions.htm).
   d. Substitute these by canoncial representations
   e. Probe some variables to detect if they are actually fixed or detect further equivalences.
4. Load the preprocessed model into the underlying SAT-solver and create the linear relaxation.
5. **Search for an optimal solution using the SAT-model (LCG) and the linear relaxation.**
6. Transform solution back to original model.

This is taken from [this talk](https://youtu.be/lmy1ddn4cyw?t=434) and slightly extended.

#### The use of linear programming techniques

As already mentioned before, CP-SAT also utilizes the (dual) simplex algorithm and linear relaxations.
The linear relaxation is implemented as a propagator and potentially executed at every node in the search tree
but only at lowest priority. A significant difference to the application of linear relaxations in branch and bound
algorithms is that only some pivot iterations are performed (to make it faster). However, as there are likely
much deeper search trees and the warm-starts are utilized, the optimal linear relaxation may still be computed, just
deeper down the tree (note that for SAT-solving, the search tree is usually traversed DFS). At root level, even
cutting planes such as Gomory-Cuts are applied to improve the linear relaxation.

The linear relaxation is used for detecting infeasibility (IPs can actually be more powerful than simple SAT, at least
in theory), finding better bounds for the objective and variables, and also for making branching decisions (using the
linear relaxation's objective and the reduced costs).

The used Relaxation Induced Neighborhood Search RINS (LNS worker), a very successful heuristic, of course also uses
linear programming.

## Further Material

Let me close this primer with some further references, that may come useful:

* You may have already been referred to the talk [Search is Dead, Long Live Proof by Peter Stuckey](TODO).
    * You can find [a recording](https://www.youtube.com/watch?v=lxiCHRFNgno) to nearly the same talk, which may be
      helpful because the slides miss the intermediate steps.
    * If you want to dig deeper, you also go
      to [the accompanying paper](https://people.eng.unimelb.edu.au/pstuckey/papers/cp09-lc.pdf).
    * This may be a little overwhelming if you are not familiar enough with SAT-solving.
* There is also [a talk of the developers of CP-SAT](https://youtu.be/lmy1ddn4cyw), which however is highly technical.
    * Gives more details on further tricks done by CP-SAT, on top of lazy clause generation.
    * The second part especially goes into details on the usage of LPs in CP-SAT. So if you are comming from that
      community, this talk will be highly interesting for you.
* The slides for the course 'Solving Hard Problems in Practice' by Jediah Katz are pretty great to understand the
  technics without any prior knowledge, however, they are currently no longer available online.
* [This blog](https://www.msoos.org/) gives some pretty nice insights into developing state of the art SAT-solvers.
* [Official Tutorial](https://developers.google.com/optimization/cp): The official tutorial is reasonably good, but
  somehow missing important information and it also seems like it is actually just updated from the previous, not so
  powerful, CP-solver.
* [Documentation](https://google.github.io/or-tools/python/ortools/sat/python/cp_model.html): The documentations give a
  good overview of the available functions but are often not extensively documented.
* [Sources](https://github.com/google/or-tools/tree/stable/ortools/sat): The sources actually contain a lot of
  information, once you know where to look. Especially a look into
  the [parameters](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto) can be very
  enlightening.
