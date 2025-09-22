# Test-Driven Development with CP-SAT

<!-- START_SKIP_FOR_README -->

![Cover Image](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/crashtest_platypus.webp)

<!-- STOP_SKIP_FOR_README -->

<a name="test-driven-optimization"></a>

> [!WARNING]
>
> This chapter is a draft, but feedback is already welcome. Current TODOs:
>
> 1. Some ideas/views Richard Oberdieck presented in
>    [this tutorial](https://github.com/RichardOberdieck/opti_test), especially
>    the idea of Hypothesis testing.
> 2. [This post by Princeton Consultants](https://princetonoptimization.com/blog/rapid-optimization-model-development-python-and-pandas-7-steps/)
>    has some strong points on teamwork and tabular data, whereas this chapter
>    focuses more on individual work and complex data structures. There should
>    be at least a reference to this post, but probably the text should also be
>    extended to at least also cover tabular data and its validation, e.g., via
>    pandera, instead of just Pydantic.

In this chapter, we demonstrate how to apply test-driven development (TDD)
principles to the nurse rostering problem using the CP-SAT solver. Our objective
is to build a model that is modular, thoroughly tested, extensible, and
straightforward to maintain.

I consider TDD a valuable approach for optimization problems, even though I do
not follow the formal TDD cycle strictly. Writing tests early helps clarify
requirements, refine problem specifications, and expose ambiguities in the
initial formulation. In many real-world scenarios, the problem statement is
incomplete or evolving. By expressing expectations as executable tests, we can
communicate requirements more effectively and adjust them as new insights
emerge.

The nurse rostering problem is particularly suitable for illustrating this
approach. It represents a class of scheduling problems where constraints and
objectives change frequently in response to stakeholder feedback. In such
settings, modular design and systematic testing substantially reduce the cost of
adapting and validating the model as requirements evolve.

For small or well-defined problems, a simple implementation with a few manual
checks may suffice. However, as complexity grows and outcomes become harder to
verify informally, a structured, test-driven approach proves highly beneficial.
It allows the problem to be decomposed into smaller, testable components that
can be incrementally implemented, validated, and refined.

This chapter does not advocate for perfection at the outset. In practice,
building a working prototype often comes first, and design improvements are
introduced once they offer clear benefits. The key is to maintain a balance
between rapid prototyping and introducing structure and tests at the right stage
of development.

We will incrementally construct a nurse rostering model using CP-SAT, guided by
test-driven principles. The workflow includes defining formal **data schemas**
for inputs and outputs, implementing **solver-independent validation functions**
that encode constraints and objectives, and creating **modular components** for
decision variables, constraints, and soft objectives. We conclude by combining
these modules into a complete solver and verifying its solutions on test
instances. However, before doing so, we will briefly review a classical approach
and what can go wrong when the problem complexity increases.

> :warning:
>
> This chapter is not intended as a strict guideline but as one possible
> approach. I do not follow every step outlined here in every project. Instead,
> I aim to share principles that I have found useful in practice. Operations
> researchers may gain insights into software engineering practices, while
> software engineers may learn how to apply their skills to optimization models.
> Experienced optimizers may find value in comparing their own approach with the
> one presented here.

---

> :video:
>
> There appears to be limited material addressing software engineering practices
> specifically in operations research. However, the talk
> [_Optimization Modeling: The Art of Not Making It an Art_](https://www.gurobi.com/events/optimization-modeling-the-art-of-not-making-it-an-art/)
> by Ronald van der Velden (Gurobi) is an excellent resource that partially
> overlaps with the themes discussed in this chapter. (Note: Registration with a
> company name and email address is required to access the content.)

You can find the complete code for this chapter
[here](https://github.com/d-krupke/cpsat-primer/tree/main/examples/tdd).

## Classical Steps for Textbook Problems

<!-- From classroom to complex reality -->

Before introducing the more advanced TDD-inspired workflow, let us briefly
review a classical approach to implementing optimization models, as commonly
encountered in textbook exercises or university homework. This traditional
approach can also be effective for many real-world problems, provided their
complexity remains moderate. However, real-world problems often introduce
additional challenges that are absent from well-structured academic exercises.
As the problem complexity increases, this classical approach becomes difficult
to manage. We will therefore explore how software engineering practices can help
address these challenges.

Let us consider a concrete example: the **Facility Location Problem (FLP)**. To
set the right context, we will present it as if it were stated directly on a
homework sheet:

<!-- Textbook Exercise FLP -->

> A delivery company is planning to open new warehouses to serve its customers.
> There is a set $F$ of potential warehouse locations, and a set $C$ of
> customers who need to receive their orders.
>
> Opening a warehouse at location $i \in F$ comes with a fixed cost $f_i$.
> Shipping goods from warehouse $i \in F$ to customer $j \in C$ costs $c_{i,j}$.
>
> Every customer $j \in C$ must be served by exactly one warehouse, and a
> warehouse can only ship goods if it is actually opened.
>
> The company needs to decide which warehouse locations $i \in F$ to open and
> which warehouse will serve each customer $j \in C$ so that the total
> cost—consisting of the opening costs $f_i$ and the shipping costs $c_{i,j}$—is
> as small as possible.

<!-- Approach to convert it into a mathematical model -->

Now, a good idea would be to highlight parameters, decisions, constraints, and
objectives in the text (using different colors), and then write down the
mathematical formulation of the problem in the following steps:

1. **Parameters:** Define the problem parameters, specifying the input data for
   the model.
2. **Decision Variables:** Define the variables representing the decisions or
   assignments to be made.
3. **Constraints and Objectives:** Specify the constraints that must be
   satisfied and the objectives to be optimized.

<!-- The mathematical model -->

For this problem, this would look like this:

> **Parameters:**
>
> - $F$: Set of potential facility locations.
> - $C$: Set of customers.
> - $f_i \in \mathbb{N}_0$ for $i \in F$: Fixed cost of opening facility $i$.
> - $c_{i,j} \in \mathbb{N}_0$ for $i \in F, j \in C$: Cost of serving customer
>   $j$ from facility $i$.
>
> **Decision Variables:**
>
> - $y_i \in \mathbb{B} \ \forall i \in F$: $y_i = 1$ if facility $i$ is opened,
>   and 0 otherwise.
> - $x_{i,j} \in \mathbb{B} \ \forall i \in F, j \in C$: $x_{i,j} = 1$ if
>   customer $j$ is served by facility $i$, and 0 otherwise.
>
> **Objective Function:**
>
> Minimize the total cost of opening facilities and serving customers:
> $\min \sum_{i \in F} f_i \cdot y_i + \sum_{i \in F} \sum_{j \in C} c_{i,j} \cdot x_{i,j}.$
>
> **Constraints:**
>
> 1. **Customer Assignment:** Each customer must be served by exactly one open
>    facility: $\sum_{i \in F} x_{i,j} = 1 \quad \forall j \in C.$
> 2. **Facility Activation:** A customer can only be served by a facility if
>    that facility is open: $x_{i,j} \leq y_i \quad \forall i \in F, j \in C.$

<!-- The mathematical model is concise and precise -->

The mathematical notation is so compact that one can easily work through it on a
whiteboard or a sheet of paper, which makes it ideal for group discussions. Such
concise, precise notation also makes it easier to identify inconsistencies or
errors directly at least for smaller problems.

<!-- implementation -->

Finally, you implement it either with your favorite optimization framework or
modelling language. In case of an exercise, it is also very likely that your
instructor has already provided some test instances, which you can use to verify
your implementation.

```python
from ortools.sat.python import cp_model

# 1. Parameters
F = [1, 2, 3]  # Set of potential facility locations.
C = [1, 2, 3]  # Set of customers.
f = {1: 100, 2: 200, 3: 150}  # Fixed cost f[i] of opening facility i.
c = {1: {1: 10, 2: 20, 3: 30},
     2: {1: 15, 2: 25, 3: 35},
     3: {1: 20, 2: 30, 3: 40}}  # Cost c[i][j] of serving customer j from facility i.

model = cp_model.CpModel()

# 2. Decision variables
y = {i: model.new_bool_var(f'y_{i}') for i in F}  # y[i] = 1 if facility i is opened.
x = {(i, j): model.new_bool_var(f'x_{i}_{j}') for i in F for j in C}  # x[i,j] = 1 if customer j is served by facility i.

# 3. Objective function
model.minimize(
    sum(f[i] * y[i] for i in F) + sum(c[i][j] * x[i, j] for i in F for j in C)
)
# 4. Constraints
for j in C:
    model.add(sum(x[i, j] for i in F) == 1)  # Each customer must be served by exactly one open facility.
for i in F:
    for j in C:
        model.add(x[i, j] <= y[i])  # A customer can only be served by a facility if that facility is open.

# Solve the model
solver = cp_model.CpSolver()
status = solver.solve(model)
# ...
```

<!-- classes instead of functions -->

Usually you would implement this in a function; however, I often need to make
models incremental, which is not compatible with stateless functions. Moreover,
using classes in Python is almost as straightforward as using pure functions.
Being able to manipulate the model after construction, like fixing certain
variables, also makes it easier to test and debug.

<!-- Example of wrapping it into a class. -->

I would encapsulate this code in a class to enhance reusability, resulting in an
implementation similar to the following:

```python
class FacilityLocationModel:
    def __init__(self, F, C, f, c):
        self.F, self.C, self.f, self.c = F, C, f, c
        self.model = cp_model.CpModel()

        # 2. Decision variables
        self.y = {i: self.model.new_bool_var(f'y_{i}') for i in self.F}  # y[i] = 1 if facility i is opened.
        # ...

    def solve(self, **parameters):
        # Solve the model
        solver = cp_model.CpSolver()
        # ...
```

<!-- you could also just directly model it in code -->

As you can see, the implementation is actually nearly identical to the
mathematical formulation and many experts may opt to directly implement it,
especially when using modeling languages like Pyomo, GAMS, or AMPL.

<!-- also works for many practical cases as many practical problems are basic -->

In practice, many cases are still simple enough to follow this workflow.
Although the data may not be readily available or may require extensive
preprocessing, and the problem statement may not yet be fully defined, it is
often possible to sketch the mathematical formulation and then implement it in a
single step. This is particularly feasible because many problems are essentially
well-known combinatorial optimization problems, merely obscured by
domain-specific terminology. With experience, you can often identify the
underlying structure of the problem quickly and adapt them to custom constraints
with ease.

<!-- perfect for simple problems -->

As long as the model remains this simple, you may only need to add a few basic
tests to verify that the solutions behave as expected; there is no need for
modularization or extensive validation functions. Such a model is easy to
understand and can be implemented within minutes. It can be presented on a
single slide, and any additional structure would likely introduce unnecessary
complexity rather than improving clarity.

<!-- from simplifying abstraction to incomprehensible, similar to tech debt -->

Mathematical abstraction can simplify a problem; however, it can also become so
abstract that the original problem is no longer recognizable. While the
threshold varies depending on one's background, there comes a point at which
monolithic mathematical formulations become incomprehensible, and even minor
modifications require significant effort. It will also be extremely difficult to
debug such a model if you cannot look at components in isolation. This situation
parallels a software architecture that, while initially straightforward, has
accumulated complexity through incremental growth, to the point where any
modification is fraught with risk and development velocity suffers.

<!-- Examples of things that can go wrong -->

Here are a few things that can go wrong when the model grows too complex:

- You are the only one who understands the model, making it unmaintainable by
  others.
- Even you no longer fully understand the model, causing simple changes to take
  forever or to break things—often only discovered when it fails in production.
- The model's complexity makes it impossible to verify individual components in
  isolation, making root-cause analysis extremely difficult.
- Testing becomes impractical, leaving you with constant doubts about whether
  the model is correct.
- A subtle bug silently excludes high-quality solutions, leading to significant
  opportunity loss without any explicit error.
- Bad or inconsistent data goes undetected due to missing validation checks,
  producing incorrect results and eroding trust in your code.
- Communication with stakeholders breaks down because you lack a clear, shared
  language to formally specify requirements.
- The model becomes so convoluted that implementing all requirements—or even
  building a first useful prototype—feels impossible.
- No clear input/output data interfaces, making collaboration and integration
  difficult.
- ...

<!-- These are failures on the "easy" parts -->

All of these issues may fail the project, before even reaching the real
challenges of optimizing scalability and ensuring real-world accuracy.

<!-- Overview -->

In the remainder of this chapter, we will explore how to avoid this situation by
applying software engineering practices to optimization models on the example of
the nurse rostering problem. We will do the following steps:

1. **Data Schema:** Define structured schemas for the problem instance and
   solution, covering parameters and parts of the decision space.
2. **Validation Functions:** Implement functions to verify feasibility and
   compute objective values, serving as the formal specification.
3. **Decision Variables:** Introduce decision variables encapsulated in
   containers to simplify constraint and objective construction.
4. **Modular Constraints and Objectives:** Build constraints and soft objectives
   as independent, testable modules.
5. **Solver Integration:** Combine these components into a complete CP-SAT model
   and test it in completion to check that the components work together as
   expected.

<!-- not all-or-nothing -->

> [!TIP]
>
> You can transition fluently between the two approaches; it is not an
> all-or-nothing choice. It is entirely feasible to adopt only parts of the
> approach described here and apply them to selected aspects of the problem.

## The Nurse Rostering Problem

Before implementing a solution, we first outline the nurse rostering problem and
its initial requirements. The
[nurse rostering problem](https://en.wikipedia.org/wiki/Nurse_scheduling_problem)
involves assigning a set of nurses to shifts over a defined planning horizon.
Each shift has a specified demand indicating the minimum number of nurses
required. The goal is to produce an assignment that satisfies all operational
constraints while optimizing soft preferences and priorities. In this example,
the requirements are as follows:

**Constraints (Hard Requirements):**

1. **Unavailability Constraint:** A nurse must not be assigned to any shift for
   which they are unavailable or explicitly blocked.
2. **Shift Coverage Requirement:** Each shift must be staffed with a sufficient
   number of nurses to meet its demand.
3. **Rest Period Constraint:** A nurse must have an adequate rest period between
   consecutive shifts. If the interval between the end of one shift and the
   start of the next is too short, the nurse cannot be assigned to both.

**Objectives (Soft Goals):**

1. **Preference Satisfaction:** Nurses may indicate preferences for particular
   shifts. The model should honor these preferences wherever possible.
2. **Staffing Preference:** Internal staff members should be preferred over
   external or contract nurses when all other constraints are satisfied.

To keep the initial formulation manageable, we start with this limited set of
constraints and objectives. The implementation is designed to be **modular and
extensible**, allowing additional requirements—such as fairness, seniority,
on-call shifts, and others—to be introduced later with minimal refactoring.
While the initial requirements may still be easily implemented in a monolithic
model, it should already be complex enough to illustrate the benefits of a more
structured, test-driven approach.

> [!WARNING]
>
> This chapter focuses on correctness, not performance. First, ensure the
> solution is correct; only then focus on making it efficient. Correspondingly,
> the implementation here may not be the most efficient.

## Instance and Solution Schema

<!-- Textbook examples assume perfect data; real-world data rarely is -->

In academic or textbook settings, one often starts with clean, well-structured
data that is perfectly tailored to the problem at hand. In real-world projects,
however, this is rarely the case. You are often lucky to receive any data at
all—let alone data that is complete, consistent, and ready for use.

<!-- Data might not even exist yet, or only on paper, or be incomplete -->

Data may not yet exist in digital form, or it may still be collected
manually—perhaps even on paper. Even if a digital system is under development,
it may not yet be deployed, and any available data might be outdated or
incomplete. Privacy concerns or legal constraints may further restrict access to
critical information, especially when working with personnel data, such as nurse
schedules.

> [!TIP]
>
> Obtain a data sample as early as possible, even if it is only an "educated
> guess" or a small subset of the actual dataset. Such samples simplify the
> extraction of test cases and support iterative discussions with stakeholders,
> who often overlook trivial but important details. For instance, the model
> developed here does not enforce a single shift per day, focusing instead on
> rest periods and total working hours over a given period (e.g., a week).
> Stakeholders are likely to recognize such omissions as soon as they review a
> solution for a realistic instance. Addressing these issues typically requires
> an agile, iterative process, which we omit here for simplicity but which can
> be applied throughout all phases of this example.

<!-- Even available data is often inconsistent or unreliable -->

And when the data does arrive, it often brings problems of its own. It may be
incomplete, contain contradictory entries, or be structured inconsistently.
Excel remains a de facto standard in many organizations, which introduces risks
such as data type mismatches, inconsistent formatting, or broken formulas. Even
worse are schemaless databases like Firestore, which offer no guarantees about
the structure or consistency of stored documents. It is not uncommon to find
fields that sometimes contain integers, sometimes strings, and occasionally just
vanish altogether—along with keys that are misspelled, duplicated, or silently
dropped.

<!-- Bad input leads to failure in your code, not just theirs -->

In such environments, any error in the data will propagate directly to your
application. Your logic fails not because your model is incorrect, but because
upstream inputs violate assumptions you did not realize you were making.

<!-- Introduce schema validation and Pydantic as solution -->

After enough frustrating encounters with malformed data, I became a strong
proponent of defining formal data schemas as a first step in any serious
optimization project. For Python, I have found
[Pydantic](https://docs.pydantic.dev/latest/) to be an excellent tool: it allows
you to define input and output schemas declaratively, enforces type checks and
invariants automatically, and raises clear validation errors when assumptions
are violated. With these schemas in place, problems can be caught early and
diagnosed easily—long before they produce incorrect results or puzzling
failures.

<!-- Schemas are not just for machines; they clarify team communication -->

Defining a schema is also crucial for ensuring that all collaborators (including
clients) are aligned in their understanding of the problem. You may all be using
the same terms—like "shift", "start time", or "duration"—but interpret them
differently. Is that duration in seconds, minutes, or hours? Is the "shift ID"
an index or a foreign key to a separate table? Without a precise schema, it is
easy to make dangerous assumptions.

<!-- Example from history: Mars Climate Orbiter -->

A vivid historical example is the loss of the
[Mars Climate Orbiter](https://en.wikipedia.org/wiki/Mars_Climate_Orbiter)
in 1999. The spacecraft was destroyed because one team used imperial units while
another expected metric, and there was no enforced contract between the systems.
The entire $327 million mission failed because of a silent mismatch in a shared
data interface. While most optimization projects are not launching spacecraft,
the takeaway is just as relevant: never assume shared understanding—make your
assumptions explicit.

<!-- Start with schemas before writing any logic -->

That is why I prefer to define and document both the input and output schemas at
the beginning of a project, even before writing the first line of algorithmic
code. It helps formalize the problem, clarifies the roles of different fields,
and allows everyone to work toward the same structure. Even if the schema
evolves over time (and it almost certainly will), the evolution is transparent,
versioned, and easy to manage.

<!-- Summary: schemas are foundational -->

Below is the data schema we will be using for the nurse rostering problem. We
made two non-trivial additions: `preferred_shift_weight` and `staff_weight`.
These will unlikely be part of the initial problem formulation, but as we have
two objectives, this would allow us to easily weight the two objectives against
each other. It could be that later on, we decide that the optimization should be
lexicographic, making these weights irrelevant, but it is a good starting point
to have at least some way of balancing the two objectives. If there are more
parameters to specify the problem, e.g., switching constraints on and off, I
would create a separate `NurseRosteringParameters` class that contains these
parameters, instead of integrating them into the instance schema, but let us
skip that for simplicity.

```python
"""
This module defines the data schema for the nurse rostering problem.
Note that this is just a random variant of the nurse rostering problem.

We define the instance and solution data structures using Pydantic.
"""

from datetime import datetime, timedelta
from pydantic import BaseModel, Field, NonNegativeInt, model_validator, PositiveInt
import uuid

# Semantic type aliases for clarity
NurseUid = int
ShiftUid = int


def generate_random_uid() -> int:
    # Use uuid4 and convert to an integer (truncated to 64 bits for practical use)
    return uuid.uuid4().int >> 64


class Nurse(BaseModel):
    """
    Represents a nurse whose shifts we want to plan. Will be part of `NurseRosteringInstance`.
    """
    uid: NurseUid = Field(
        default_factory=generate_random_uid,
        description="Unique identifier for the nurse",
    )
    name: str = Field(..., description="Name of the nurse")
    preferred_shifts: set[ShiftUid] = Field(
        ..., description="List of preferred shift UIDs for the nurse"
    )
    blocked_shifts: set[ShiftUid] = Field(
        ..., description="List of blocked shift UIDs for the nurse"
    )
    staff: bool = Field(
        ...,
        description="Indicates if the nurse is a staff member (True) or a contractor (False)",
    )
    min_time_between_shifts: timedelta = Field(
        ..., description="Minimum off duty time between two shifts for the same nurse"
    )
    preferred_shift_weight: NonNegativeInt = Field(
        default=1,
        description="The weight in the objective function for every assigned preference.",
    )


class Shift(BaseModel):
    """
    Represents a shift that needs to be covered by nurses. Will be part of `NurseRosteringInstance`.
    """
    uid: ShiftUid = Field(
        default_factory=generate_random_uid,
        description="Unique identifier for the shift",
    )
    name: str = Field(..., description="Name of the shift (e.g., '2025-01-01 Morning')")
    start_time: datetime = Field(
        ..., description="Start time of the shift as a full datetime (YYYY-MM-DD HH:MM)"
    )
    end_time: datetime = Field(..., description="End time of the shift as a full datetime (YYYY-MM-DD HH:MM)")
    demand: NonNegativeInt = Field(
        ..., description="Number of nurses required for this shift"
    )

class NurseRosteringInstance(BaseModel):
    """
    This schema defines the INPUT for the nurse rostering problem.
    """
    nurses: list[Nurse] = Field(
        ..., description="List of nurses in the rostering instance"
    )
    shifts: list[Shift] = Field(
        ...,
        description="List of shifts that need to be covered. Shifts must be sorted in time.",
    )
    staff_weight: int = Field(
        default=1,
        description="The weight in the objective function for each assigned staff nurse.",
    )

    @model_validator(mode="after")
    def validate_shifts_unique_uids(self):
        """
        Ensure that all shifts have unique UIDs to avoid conflicts.
        """
        shift_uids = {shift.uid for shift in self.shifts}
        if len(shift_uids) != len(self.shifts):
            raise ValueError("Shift UIDs must be unique.")
        return self

    @model_validator(mode="after")
    def validate_nurses_unique_uids(self):
        """
        Ensure that all nurses have unique UIDs to avoid conflicts.
        """
        nurse_uids = {nurse.uid for nurse in self.nurses}
        if len(nurse_uids) != len(self.nurses):
            raise ValueError("Nurse UIDs must be unique.")
        return self

    @model_validator(mode="after")
    def validate_shifts_sorted_by_time(self):
        """
        Ensure that shifts are sorted by start time.
        """
        for shift_a, shift_b in zip(self.shifts, self.shifts[1:]):
            if shift_a.start_time > shift_b.start_time:
                raise ValueError("Shifts must be sorted by start time.")
        return self
```

<!-- Solution schema and pydantic vs native lists -->

For the solution, each shift is mapped to a list of assigned nurses. In
addition, the objective value and the timestamp at which the solution was
computed are recorded. As a potential improvement, the `list[NurseUid]`
structure could be replaced with a dedicated Pydantic model. This approach would
allow for the inclusion of additional shift-specific data in the future, which
cannot be directly attached to a plain `list`. Such a design would be
particularly advantageous when the solution must be presented in a user
interface. Nevertheless, as long as the base object is defined as a Pydantic
model, it is generally possible to maintain backward compatibility when
extending the schema, even when certain native Python elements are involved.

```python
class NurseRosteringSolution(BaseModel):
    """
    This schema defines the OUTPUT for the nurse rostering problem.
    """
    nurses_at_shifts: dict[ShiftUid, list[NurseUid]] = Field(
        ..., description="Maps shift UIDs to lists of assigned nurse UIDs."
    )
    objective_value: int = Field(
        description="Objective value of the computed solution."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Time when the solution was generated. Takes little space and can be extremely useful when investigating issues with the solution. Optimally, also add the revision of the algorithm that generated the solution, e.g., by using a git commit hash.",
    )
    # Validation of the solution will be handled in a separate module.
```

With these schemas in place, it becomes immediately clear what structure the
input and output must follow. The `model_validator` methods let you encode
important assumptions—such as uniqueness or sorting—and they raise explicit
errors if those assumptions are violated. While this does not guarantee that all
data is actually correct, it prevents many accidental errors and makes debugging
significantly easier. By catching issues at the boundary, you gain confidence
that the optimization logic operates on well-formed inputs.

> [!TIP]
>
> You do not need to use the same schema for the endpoint contract and the
> internal representation of the optimization model. Converting between
> different schemas is inexpensive compared to the computational cost of solving
> the model. If a different representation enables more efficient model
> construction, introduce an additional layer. Remember that the internal schema
> can be modified easily, but changes to the outward-facing schema must be
> handled carefully to avoid breaking existing interfaces.

### Tabular Data

<!-- Tabular data is common and very nice to work with, if it is clean -->

Many people, including myself, appreciate working with tables. In the chapter’s
example project, we use a nested data structure. However, in many projects
tabular data is entirely sufficient, and in such cases you should prefer this
representation. In principle, any data structure can be expressed in tabular
form, and most common databases, in particular relational databases accessed via
SQL, are built around tables. Nevertheless, when extracting information requires
complex joins or aggregations, you may need to either restructure your tables or
adopt a more object-oriented format, for example by using Pydantic models as
shown above. Since input data should not be modified during model construction,
it is perfectly acceptable to employ both representations in parallel.

If your data can be naturally and intuitively expressed in tabular form, then
you should retain this representation. Python provides excellent libraries for
working with tabular data, most notably [Pandas](https://pandas.pydata.org/) and
[Polars](https://www.pola.rs/). Pandas is the most widely used library for data
analysis in Python, and it benefits from a rich ecosystem of extensions and
integrations. However, it can exhibit performance limitations, particularly with
large datasets. Polars is a newer library that is more performance-oriented.
Since optimization problems typically involve much smaller datasets than those
encountered in data analysis or machine learning, the widespread use and
ecosystem of Pandas usually make it the more suitable choice.

The major advantage of storing data in a Pandas DataFrame is the ability to
perform aggregations, filtering, and transformations with a concise and
expressive syntax. In addition, Pandas integrates well with visualization
libraries such as [Matplotlib](https://matplotlib.org/) and
[Seaborn](https://seaborn.pydata.org/), which allow you to easily create
informative visual representations of your data.

Do you want to scale a column to the range [0, 1]? This requires only a single
line of code:

```python
df['scaled_column'] = (df['column'] - df['column'].min()) / (df['column'].max() - df['column'].min())
```

Do you want to filter rows based on a condition? Again, this can be achieved in
one line:

```python
filtered_df = df[df['column'] > threshold]
```

CP-SAT also provides integration with Pandas, as demonstrated in the following
knapsack example, where the values and weights of items are stored in a
`DataFrame`:

```python
from ortools.sat.python import cp_model
import pandas as pd

# Example data: each row is an item with a weight and a value
# In production, you would likely load this from a file or database,
# which pandas has simple loaders for.
df = pd.DataFrame(
    data={
        "weight": [2, 4, 3, 5, 1, 6, 2, 7, 3, 4],
        "value": [10, 8, 7, 6, 5, 9, 4, 3, 2, 1],
    },
    index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
)

model = cp_model.CpModel()

# Create a Boolean variable for each item (indexed by DataFrame index)
x = model.new_bool_var_series("x", df.index)

# Use Pandas vectorized operations to build constraints and objectives
model.add((x @ df["weight"]) <= 15)  # total weight limit
model.maximize(x @ df["value"])      # maximize total value
```

Some constraints can also be conveniently added row by row. For example, suppose
we have a set of prohibited pairs of items that cannot be selected together.
Such restrictions can be represented naturally as a two-column table:

```python
prohibited_pairs_df = pd.DataFrame(
    data={
        "item1": ["A", "B", "C"],
        "item2": ["D", "E", "F"],
    }
)

# For each row, apply the corresponding constraint
for _, row in prohibited_pairs_df.iterrows():
    model.add(x[row["item1"]] + x[row["item2"]] <= 1)
```

Suppose we have not only prohibited pairs of items but arbitrary sets of items
that cannot be selected together. To express this structure in a classical
tabular format, we would introduce a unique identifier for each set and a
relation table that maps item identifiers to set identifiers. To build the
constraints, we would first group the rows by set identifier, aggregate the
corresponding item identifiers, and then add the constraint for each set. If
this requirement constitutes only a small portion of the model, the tabular
approach is acceptable. However, if the model relies on many such sets or lists,
I recommend switching to a more object-oriented representation, as used in the
nurse rostering problem in this chapter.

When working with tabular data, it is still important to enforce a strict
schema, as we did with the Pydantic models above. This can be achieved with
libraries such as [pandera](https://pandera.readthedocs.io/en/stable/). For
instance, we can require that the weights and values of the items are
non-negative integers. Although this may appear trivial, errors can easily
arise, for example when data is extracted from widely used Excel files somewhere
in the processing pipeline.

```python
import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "weight": pa.Column(int, pa.Check.ge(0)),
    "value": pa.Column(int, pa.Check.ge(0)),
})

validated_df = schema.validate(df)
```

> :reference:
>
> Princeton Consultants provide an excellent post on
> [Rapid Optimization Model Development with Python and pandas in 7 Steps](https://princetonoptimization.com/blog/rapid-optimization-model-development-python-and-pandas-7-steps/),
> which offers a more detailed discussion of the model-building process with
> Pandas. In my own work, I am often involved in projects where the data is more
> graph-like or hierarchical, in which case Pydantic tends to be a more natural
> representation for building optimization models. I explicitly added this
> subsection to emphasize that using nested data with Pydantic is not the only
> approach, and it is not always the best one. Ultimately, you should choose the
> representation that best matches your data and your team. If necessary, use
> multiple representations in parallel, as the additional memory overhead is
> usually negligible.

## Solver-Agnostic Validation

<!-- Writing validation functions early clarifies the constraints and enables TDD -->

With the data schemas defined, we can now implement validation functions that
check whether a proposed solution satisfies all constraints. This is the first
concrete step toward a test-driven workflow.

<!-- Emphasizes the value of validation for synchronization with stakeholders -->

These validation functions are solver-independent, meaning they do not rely on
any specific algorithm or library like CP-SAT. This makes them ideal for
communicating and verifying the specification with domain experts or clients. In
fact, someone with only basic programming skills could review or even contribute
to these checks, reducing misunderstandings about the problem's requirements.

<!-- Clarifies that defining the objective function explicitly is a big step forward -->

In addition to feasibility, we also define the objective function here. That
might seem unusual for validation logic, but it provides a clear and unambiguous
specification of what we are trying to optimize. In practice, the objective
function is often the most volatile part of the specification—frequently revised
as new goals, costs, or incentives are introduced. By expressing it in code, we
make it easier to reason about and iterate on collaboratively.

> :warning:
>
> Crafting an effective objective function requires experience. I have often
> observed less experienced practitioners reaching incorrect conclusions about
> its proper formulation. Common errors include unintentionally encoding parts
> of the objective as constraints or defining the objective in a way that allows
> undesirable trade-offs between its components. If a client provides an
> objective function, do not implement it blindly; instead, analyze it carefully
> before proceeding.

<!-- Summarizes the benefits -->

By writing these checks up front, we make the problem precise and executable.
Even without an optimization engine, we already know how to distinguish valid
from invalid solutions and how to compare two feasible solutions in terms of
quality. This gives us a solid foundation for developing and testing
optimization algorithms in the next steps. If we are lucky, we could even use it
as a feedback loop for an AI assistant to code the solution for us, although my
experiences with that have been mixed so far.

The following Python module provides these validation checks and a function to
compute the objective value:

```python
"""
This module contains validation functions for nurse rostering solutions.
It is solver and algorithm agnostic, meaning it can be used to validate solutions
from any solver that produces a `NurseRosteringSolution` object. Just by providing
such functions, you have taken a huge step toward computing a solution, as you now
have a clean specification of what a valid and good solution looks like.
"""

from collections import defaultdict
from .data_schema import NurseRosteringInstance, NurseRosteringSolution


def assert_consistent_uids(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
):
    """
    Assert that all UIDs in the solution are part of the instance.
    """
    nurse_uids = {n.uid for n in instance.nurses}
    shift_uids = {s.uid for s in instance.shifts}
    for shift_uid, nurse_list in solution.nurses_at_shifts.items():
        if shift_uid not in shift_uids:
            raise AssertionError(f"Shift {shift_uid} is not present in the instance.")
        for nurse_uid in nurse_list:
            if nurse_uid not in nurse_uids:
                raise AssertionError(
                    f"Nurse {nurse_uid} is not present in the instance."
                )


def assert_no_blocked_shifts(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
):
    """
    Assert that no nurse is assigned to a blocked shift in the solution.
    """
    for nurse in instance.nurses:
        for shift_uid in nurse.blocked_shifts:
            if (
                shift_uid in solution.nurses_at_shifts
                and nurse.uid in solution.nurses_at_shifts[shift_uid]
            ):
                raise AssertionError(
                    f"Nurse {nurse.uid} is assigned to blocked shift {shift_uid}."
                )


def assert_demand_satisfaction(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
):
    """
    Assert that each shift meets its nurse demand.
    """
    for shift in instance.shifts:
        assigned = solution.nurses_at_shifts.get(shift.uid, [])
        if len(assigned) < shift.demand:
            raise AssertionError(
                f"Shift {shift.uid} demand not met: {len(assigned)}/{shift.demand} assigned."
            )


def assert_min_time_between_shifts(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
):
    """
    Assert that nurses are not assigned to shifts too close together.
    """
    shifts_by_uid = {s.uid: s for s in instance.shifts}
    nurse_to_shifts = defaultdict(list)
    for shift_uid, nurse_uids in solution.nurses_at_shifts.items():
        for nurse_uid in nurse_uids:
            nurse_to_shifts[nurse_uid].append(shifts_by_uid[shift_uid])
    for nurse in instance.nurses:
        assigned = sorted(nurse_to_shifts[nurse.uid], key=lambda s: s.start_time)
        for a, b in zip(assigned, assigned[1:]):
            if b.start_time < a.end_time + nurse.min_time_between_shifts:
                raise AssertionError(
                    f"Nurse {nurse.uid} assigned to shifts {a.uid} and {b.uid} with insufficient rest."
                )


def objective_value(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
) -> int:
    """
    Calculate the objective value of the solution based on the instance's preferences and staff assignments.
    """
    nurses_by_uid = {n.uid: n for n in instance.nurses}
    obj_val = 0
    for shift_uid, nurse_uids in solution.nurses_at_shifts.items():
        for nurse_uid in nurse_uids:
            nurse = nurses_by_uid[nurse_uid]
            if shift_uid in nurse.preferred_shifts:
                # Preferred shifts decrease the objective (we minimize).
                obj_val -= nurse.preferred_shift_weight
            if not nurse.staff:
                # Non-staff nurses incur a penalty (they are more expensive).
                obj_val += instance.staff_weight
    return obj_val


def assert_solution_is_feasible(
    instance: NurseRosteringInstance,
    solution: NurseRosteringSolution,
    check_objective: bool = True,
):
    """
    Run all standard feasibility checks.
    """
    assert_consistent_uids(instance, solution)
    assert_no_blocked_shifts(instance, solution)
    assert_demand_satisfaction(instance, solution)
    assert_min_time_between_shifts(instance, solution)
    if check_objective:
        obj_val = objective_value(instance, solution)
        if obj_val != solution.objective_value:
            raise AssertionError(
                f"Objective value mismatch: expected {obj_val}, got {solution.objective_value}."
            )
```

<!-- validation functions are solver-independent and prioritize clarity -->

The main advantage of these validation functions is that they are completely
independent of any solver or optimization model. This means we do not need to
optimize their performance—they are meant to be correct, clear, and easy to
understand. Precise and informative error messages are essential, as these will
guide you when tests fail.

<!-- validation is especially valuable for complex problems -->

For simple problems, this level of validation may appear excessive, especially
since CP-SAT already uses a declarative modeling style. However, these functions
provide an independent specification of the problem, which is valuable for
catching misunderstandings and hidden assumptions early.

<!-- validation and CP-SAT complement each other -->

For more complex problems, these validation functions are often simpler and more
intuitive than the corresponding CP-SAT constraints. For example, when enforcing
minimum rest times between shifts, the validation logic directly checks for
violations, whereas CP-SAT encodes permissible patterns of assignments. In that
sense, validation and CP-SAT complement each other: one detects what is wrong,
the other defines what is allowed.

<!-- allows you to completely change the optimization algorithm -->

Lastly, it is quite possible that a pure CP-SAT model will not scale
sufficiently, despite all optimizations and you will have to decompose the
problem. These validation functions would remain useful and safeguard you while,
e.g., replacing the vanilla CP-SAT model with a large neighborhood search or a
metaheuristic approach.

> [!TIP]
>
> You could also add unit tests for these validation functions to ensure that
> they raise the expected errors for invalid inputs. However, I would consider
> this unnecessary in most cases, as the complementary nature of validation
> functions and CP-SAT constraints will likely catch most inconsistencies. If a
> test fails, always verify whether the test itself might be incorrect.

## Decision Variables

<!-- Transition from specification to modeling -->

Up to this point, we have defined the **parameters** of the problem (through our
Pydantic schemas) and created **validation functions** that serve as a
solver-independent specification of the constraints and objectives. These
validation functions essentially describe what a valid solution must look like
and how we measure its quality.

<!-- Motivation for CP-SAT model -->

The next step is to turn this high-level specification into a formal
optimization model using CP-SAT. To do so, we start by defining **decision
variables**, which represent the core choices of our problem. For the nurse
rostering problem, a natural decision variable is:

> **Is nurse $i$ assigned to shift $j$?**

This can be represented by a Boolean variable $x_{i,j}$ that is true if nurse
$i$ works shift $j$, and false otherwise.

<!-- Variable container as a design pattern -->

Instead of managing these variables as a raw two-dimensional array, we structure
them using **variable containers**. For each nurse, we create a container that
holds their assignment variables and provides useful methods for counting
shifts, iterating over assignments, and extracting the solution. This approach
keeps the model modular, makes constraints easier to implement, and supports
reusability when we later add new constraints or objectives.

```python
"""
This module provides a basic container to manage the variables for a single nurse in the nurse rostering problem.
"""

from collections.abc import Iterable
from ortools.sat.python import cp_model
from .data_schema import Nurse, Shift, ShiftUid


class NurseDecisionVars:
    """
    A container to create and manage the decision variables for a single nurse.

    Each nurse has one Boolean variable for each shift, indicating whether the nurse is assigned to that shift.
    This class also provides helper methods to iterate over assignments and extract results.
    """

    def __init__(self, nurse: Nurse, shifts: list[Shift], model: cp_model.CpModel):
        self.nurse = nurse
        self.shifts = shifts
        self.model = model
        # Create one Boolean decision variable per shift for this nurse
        self._x = {
            shift.uid: model.new_bool_var(f"assign_{nurse.uid}_{shift.uid}")
            for shift in shifts
        }

    def fix(self, shift_uid: ShiftUid, value: bool):
        """
        Fix the assignment variable for the given shift UID to a specific value (True or False).
        Useful for setting hard constraints or testing the model.
        """
        if shift_uid not in self._x:
            raise ValueError(
                f"Shift UID {shift_uid} not found in nurse {self.nurse.uid} assignments."
            )
        self.model.add(self._x[shift_uid] == value)

    def is_assigned_to(self, shift_uid: ShiftUid) -> cp_model.BoolVarT:
        """
        Return the decision variable for the given shift UID.
        This variable is True if the nurse is assigned to that shift, and False otherwise.
        """
        return self._x[shift_uid]

    def iter_shifts(self) -> Iterable[tuple[Shift, cp_model.BoolVarT]]:
        """
        Iterate over all (shift, variable) pairs for this nurse.
        """
        for shift in self.shifts:
            yield shift, self.is_assigned_to(shift_uid=shift.uid)

    def extract(self, solver: cp_model.CpSolver) -> list[ShiftUid]:
        """
        Extract a list of shift UIDs that this nurse is assigned to in the solution.
        """
        return [shift_uid for shift_uid in self._x if solver.value(self._x[shift_uid])]
```

<!-- Allows refactoring and performance hacks -->

By encapsulating the decision variables in this manner, the process of
constructing constraints and objectives is simplified. Each constraint module
can focus on a single nurse or a subset of shifts without dealing with the
low-level details of variable creation or indexing. The behavior of the
`NurseDecisionVars` class can later be modified, for example, to avoid creating
variables for shifts irrelevant to a given nurse and instead return a constant
`0` when such shifts are queried. These shifts can also be skipped in
`iter_shifts` to improve loop efficiency. Such refactoring is straightforward
with this container-based design but would be considerably more complex when
using, for instance, a raw two-dimensional array.

## Modules

<!-- Introduce modular design -->

Now that we have defined the decision variables, the next step is to translate
the problem constraints and objectives into an optimization model. For the nurse
rostering problem, we have multiple constraints that are logically independent.
Instead of writing them directly into a monolithic solver script, we will
implement each constraint or objective as a separate **module**.

<!-- Why modularity is valuable -->

This modular approach offers several benefits:

- It keeps the model organized and readable.
- Each module can be tested individually (aligned with our TDD approach).
- Adding, removing, or modifying a constraint or objective later becomes
  trivial—simply add or replace a module.

<!-- Abstract base class introduction -->

To ensure all modules follow the same structure, we define an **abstract base
class** that specifies the interface for implementing constraints and
objectives. This base class defines a single method, `build`, which applies the
module’s constraints and optionally returns a sub-objective expression. These
sub-objectives will later be combined to form the global objective function.

<!-- Linking to decision variables -->

The `build` method takes three arguments:

1. The **instance**, which contains the problem parameters.
2. The **CP-SAT model**, which we modify by adding constraints.
3. A list of **decision variable containers** (one per nurse), which gives easy
   access to all variables related to that nurse.

<!-- Code explanation -->

Here is the abstract base class for all modules:

```python
import abc
from ortools.sat.python import cp_model
from .data_schema import NurseRosteringInstance
from .nurse_vars import NurseDecisionVars


class ShiftAssignmentModule(abc.ABC):
    @abc.abstractmethod
    def build(
        self,
        instance: NurseRosteringInstance,
        model: cp_model.CpModel,
        nurse_shift_vars: list[NurseDecisionVars],
    ) -> cp_model.LinearExprT:
        """
        Apply the constraints and return an optional sub-objective expression.
        Subclasses must implement this method to define their specific constraints and objectives.
        """
        return 0
```

For testing, we will use a few auxiliary helper classes that significantly
reduce boilerplate code when validating CP-SAT models. These helpers provide an
intuitive way to assert whether a model is feasible or infeasible after
constraints have been added.

You can install these helpers via `pip install cpsat-utils`

```python
from cpsat_utils.testing import (
    AssertModelFeasible,
    AssertModelInfeasible,
    assert_objective,
)

with AssertModelFeasible() as model:
    # build a model that is supposed to be feasible
    # if the model is infeasible, the context manager will raise an error

with AssertModelInfeasible() as model:
    # build a model that is supposed to be infeasible
    # if the model is feasible, the context manager will raise an error

# assert that the model is feasible and has the expected objective value
assert_objective(model=model, expected=-1.0)
```

These helpers are particularly useful for **test-driven development (TDD)**
because they keep the tests concise and focused on **what should happen**,
rather than on the mechanics of solving and checking the model. By wrapping the
solver logic inside a context manager, you avoid repetitive setup code and make
tests easier to read and maintain.

We will write tests using [**pytest**](https://docs.pytest.org/en/stable/), a
widely used testing framework for Python. With pytest, you simply write
functions that start with `test_`, and the framework will automatically discover
and execute them. To run all tests, just type `pytest` in your terminal, and it
will find all test files (named either `test_*.py` or `*_test.py`) and run the
contained tests.

Here is a minimal example:

```python
# ./tests/test_example.py

def test_example():
    """
    This is a simple example test that always fails.
    Use it to verify that your testing framework is set up correctly.
    """
    assert False  # Make sure you get an error here to confirm the test framework is working.
```

In order to get some simple test data, we will also write some simple data
generation functions.

1. **create_shifts**: Generates a list of shifts with consecutive start times.
2. **create_nurse**: Creates a nurse with a specified name and blocked shifts.

The code is quite simple and you can find it below:

<details><summary>Click to expand the code for data generation</summary>

```python
# ./tests/generate.py
"""
A simple utility to create shifts and nurses for testing purposes.
"""

from datetime import datetime, timedelta
from nurserostering.data_schema import Shift, Nurse


def create_shifts(k: int, week: int = 0, shift_length: int = 8) -> list[Shift]:
    """
    Create a list of shifts for testing.
    Each shift is named "Shift {i}" and has a start time and end time.
    """
    shifts = []
    for i in range(k):
        start_time = (
            datetime(2025, 1, 1, 0, 0)
            + timedelta(hours=i * shift_length)
            + timedelta(days=week * 7)
        )
        end_time = (
            datetime(2025, 1, 1, 0, 0)
            + timedelta(hours=(i + 1) * shift_length)
            + timedelta(days=week * 7)
        )
        shifts.append(
            Shift(
                name=f"Shift {i + 1}",
                start_time=start_time,
                end_time=end_time,
                demand=2,
            )
        )
    return shifts


def create_nurse(
    nurse_name: str = "Test Nurse",
    preferred_shifts: set[int] | None = None,
    blocked_shifts: set[int] | None = None,
    staff: bool = True,
    min_time_between_shifts: timedelta = timedelta(hours=8),
) -> Nurse:
    """
    Create a nurse with customizable attributes.
    """
    if preferred_shifts is None:
        preferred_shifts = set()
    if blocked_shifts is None:
        blocked_shifts = set()
    return Nurse(
        name=nurse_name,
        preferred_shifts=preferred_shifts,
        blocked_shifts=blocked_shifts,
        staff=staff,
        min_time_between_shifts=min_time_between_shifts,
    )
```

</details>

### No Blocked Shifts

When we begin implementing constraints, it is wise to start with the simplest
rule possible. For our nurse rostering problem, one such rule is that **a nurse
cannot be assigned to a shift if they are explicitly blocked from working that
shift.** This models situations such as vacations, sick leave, or other forms of
unavailability. If this constraint is violated, the schedule is invalid, no
matter how good it is otherwise.

We will now apply a test-driven approach to implement this constraint.

#### Step 1: Writing the First Test (Trivial Feasibility)

Our first test checks the trivial case: a nurse with **no blocked shifts**
should not cause any infeasibility. We do not even need to fix any assignments
in this case. The purpose is to verify that the model remains valid when there
are no constraints to enforce.

```python
def test_no_blocked_shifts_trivial():
    """
    A nurse with no blocked shifts should always lead to a feasible model.
    """
    shifts = create_shifts(2)  # two consecutive shifts
    nurse = create_nurse("Nurse A", blocked_shifts=set())
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)

    with AssertModelFeasible() as model:
        nurse_vars = NurseDecisionVars(nurse, shifts, model)
        NoBlockedShiftsModule().build(instance, model, [nurse_vars])
```

This test will of course fail because we have not yet implemented the
`NoBlockedShiftsModule`. Let us do that next.

```python
class NoBlockedShiftsModule(ShiftAssignmentModule):
    """
    Prohibit assignment to blocked shifts.
    """

    def build(
        self,
        instance: NurseRosteringInstance,
        model: cp_model.CpModel,
        nurse_shift_vars: list[NurseDecisionVars],
    ) -> cp_model.LinearExprT:
        # TODO
        return 0
```

Now it should pass, as we actually do not enforce any constraints yet. Let us
create a second test that will check that we cannot assign a nurse to a shift
they are blocked from.

#### Step 2: Testing Infeasibility for Blocked Shifts

If a nurse is **blocked from working a shift** but we force them to be assigned
to it, the model should become infeasible. This confirms that our constraint
actually prevents invalid assignments.

```python
def test_no_blocked_shifts_infeasible():
    """
    The model should be infeasible if we assign a nurse to a blocked shift.
    """
    shifts = create_shifts(2)
    nurse = create_nurse("Nurse A", blocked_shifts={shifts[0].uid})
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)

    with AssertModelInfeasible() as model:
        nurse_vars = NurseDecisionVars(nurse, shifts, model)
        NoBlockedShiftsModule().build(instance, model, [nurse_vars])
        nurse_vars.fix(shifts[0].uid, True)  # Force assignment to a blocked shift
```

This test should now fail, as we have not yet implemented the logic to enforce
the blocked shifts constraint. Let us implement the logic in the
`NoBlockedShiftsModule`.

```python
class NoBlockedShiftsModule(ShiftAssignmentModule):
    """
    Prohibit assignment to blocked shifts.
    """

    def enforce_for_nurse(self, model: cp_model.CpModel, nurse_x: NurseDecisionVars):
        for shift_uid in nurse_x.nurse.blocked_shifts:
            # Prohibit assignment to blocked shifts
            model.add(nurse_x.is_assigned_to(shift_uid=shift_uid) == 0)

    def build(
        self,
        instance: NurseRosteringInstance,
        model: cp_model.CpModel,
        nurse_shift_vars: list[NurseDecisionVars],
    ) -> cp_model.LinearExprT:
        for nurse_x in nurse_shift_vars:
            self.enforce_for_nurse(model, nurse_x)
        return 0  # no objective contribution
```

#### Step 3: Testing Feasibility on Non-Blocked Shifts

Finally, we check that a nurse can still work on **other, non-blocked shifts**.
Here we assign the nurse to a shift that is allowed, and we expect the model to
remain feasible.

```python
def test_no_blocked_shifts_feasible():
    """
    A nurse assigned to a non-blocked shift should remain feasible.
    """
    shifts = create_shifts(2)
    nurse = create_nurse("Nurse A", blocked_shifts={shifts[0].uid})
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)

    with AssertModelFeasible() as model:
        nurse_vars = NurseDecisionVars(nurse, shifts, model)
        NoBlockedShiftsModule().build(instance, model, [nurse_vars])
        nurse_vars.fix(shifts[1].uid, True)  # Assign allowed shift
```

Our `NoBlockedShiftsModule` should cover that case as well. In the following
modules, we will not perform the TDD cycle in detail, but we will still describe
the key tests and the implementation of each module. This will keep the focus on
the constraints and objectives rather than the testing mechanics.

> [!TIP]
>
> How do you find good test cases? A good rule of thumb is to start with simple
> cases that cover the basic functionality. Then add edge cases from both sides:
> create one case that is right on the edge of feasibility, and another that is
> just outside it. Any scenario that could be off by one is an excellent
> candidate for an edge case.

### Minimum Off Time

The **minimum off time constraint** enforces that a nurse cannot work two shifts
that are too close together. Formally, for each nurse and for every pair of
shifts, if the time between the end of one shift and the start of the next is
less than the required rest period, both cannot be assigned to the nurse.

#### Key Tests

To verify this constraint, we first test with simple assignment patterns:

- `[None, True, True, None]`: infeasible (two consecutive shifts without rest).
- `[True, False, True]`: feasible if the gap is long enough.
- `[True, True]`: infeasible if two consecutive shifts overlap or violate the
  minimum rest.

Using a helper function, we can easily run these tests with different
parameters:

```python
def run_min_rest_test(
    assignments: list[bool|None],
    expected_feasible: bool,
    shift_length: int = 8,
    min_time_in_between: timedelta = timedelta(hours=16),
):
    shifts = create_shifts(len(assignments), shift_length=shift_length)
    nurse = create_nurse("Nurse A", min_time_between_shifts=min_time_in_between)
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)

    context = AssertModelFeasible() if expected_feasible else AssertModelInfeasible()
    with context as model:
        nurse_vars = NurseDecisionVars(nurse, shifts, model)
        MinTimeBetweenShifts().build(instance, model, [nurse_vars])
        for s, assign in zip(shifts, assignments):
            if assign is None:
                continue  # skip free assignments
            nurse_vars.fix(s.uid, assign)
```

The individual tests can then be defined as follows:

```python
def test_min_time_between_shifts_infeasible_pattern():
    run_min_rest_test(
        assignments=[None, True, True, None],
        expected_feasible=False,
        shift_length=8,
        min_time_in_between=timedelta(hours=8),
    )
```

```python
def test_min_time_between_shifts_feasible_pattern():
    run_min_rest_test(
        assignments=[True, False, True],
        expected_feasible=True,
        shift_length=8,
        min_time_in_between=timedelta(hours=8),
    )
```

Additional test cases (e.g., single-shift, all-false, and variable-length gaps)
are available in the full test file but omitted here for clarity.

> [!WARNING]
>
> For testing feasibility, you should fix as many assignments as possible, as
> each fixed assignment makes it harder for the solver to maneuver around the
> constraints. For testing infeasibility, you should fix as few assignments as
> necessary, allowing the solver enough freedom to explore potential ways of
> circumventing the constraints.

#### Implementation

For the implementation, we find for each shift of a nurse all subsequent shifts
that would violate the minimum rest time if both were assigned. If the first
shift is assigned, then none of these subsequent shifts can be assigned. Here we
can optimize the model building by leveraging the fact that the shifts are
sorted by start time.

A simpler, but less efficient, implementation would check all pairs of shifts
for each nurse, and enforce that only one of them can be assigned.

```python
class MinTimeBetweenShifts(ShiftAssignmentModule):
    def enforce_for_nurse(self, model: cp_model.CpModel, nurse_x: NurseDecisionVars):
        min_time_between_shifts = nurse_x.nurse.min_time_between_shifts
        for i in range(len(nurse_x.shifts) - 1):
            shift_i = nurse_x.shifts[i]
            colliding: list[Shift] = []  # shifts that are too close to shift_i
            for j in range(i + 1, len(nurse_x.shifts)):
                shift_j = nurse_x.shifts[j]
                if shift_i.end_time + min_time_between_shifts <= shift_j.start_time:
                    # Since shifts are sorted by start time, if the current shift_j starts
                    # after the required rest period, all subsequent shifts will also be valid.
                    # Therefore, we can safely break here to avoid unnecessary checks.
                    break
                colliding.append(shift_j)
            if colliding:
                # if there are shifts that are too close to shift_i,
                # prevent their assignment if shift_i is assigned
                shift_i_selected = nurse_x.is_assigned_to(shift_i.uid)
                no_colliding_selected = (
                    sum(nurse_x.is_assigned_to(s.uid) for s in colliding) == 0
                )
                model.add(no_colliding_selected).only_enforce_if(shift_i_selected)

    def build(self, instance, model, nurse_shift_vars):
        """
        Enforce minimum rest time between any two shifts for a nurse.
        """
        for nv in nurse_shift_vars:
            self.enforce_for_nurse(model, nv)
        return 0  # no objective contribution
```

### Demand Satisfaction

In the real world, every shift requires a minimum number of nurses to ensure
safe and efficient operations. This requirement is captured as the **demand** of
the shift. The demand satisfaction constraint enforces that each shift is
assigned at least as many nurses as its demand specifies.

To verify this behavior, we use simple tests. In the first test, we create one
shift with a demand of 2 and two available nurses, which is feasible. In the
second test, we block the assignment of one of the nurses, which violates the
demand requirement and should make the model infeasible.

```python
from nurserostering.modules import DemandSatisfactionModule
from nurserostering.nurse_vars import NurseDecisionVars
from cpsat_utils.testing import AssertModelFeasible, AssertModelInfeasible
from generate import create_shifts, create_nurse
from nurserostering.data_schema import NurseRosteringInstance

def test_demand_satisfaction_met():
    shifts = create_shifts(1)
    shifts[0].demand = 2
    nurse1 = create_nurse("N1")
    nurse2 = create_nurse("N2")
    instance = NurseRosteringInstance(nurses=[nurse1, nurse2], shifts=shifts)

    with AssertModelFeasible() as model:
        nurse_vars1 = NurseDecisionVars(nurse1, shifts, model)
        nurse_vars2 = NurseDecisionVars(nurse2, shifts, model)
        DemandSatisfactionModule().build(instance, model, [nurse_vars1, nurse_vars2])


def test_demand_satisfaction_understaffed():
    shifts = create_shifts(1)
    shifts[0].demand = 2
    nurse1 = create_nurse("N1")
    nurse2 = create_nurse("N2")
    instance = NurseRosteringInstance(nurses=[nurse1, nurse2], shifts=shifts)

    with AssertModelInfeasible() as model:
        nurse_vars1 = NurseDecisionVars(nurse1, shifts, model)
        nurse_vars2 = NurseDecisionVars(nurse2, shifts, model)
        DemandSatisfactionModule().build(instance, model, [nurse_vars1, nurse_vars2])
        nurse_vars2.fix(shifts[0].uid, False)
```

The corresponding implementation of the constraint is equally straightforward:
we simply add a constraint that the total number of assigned nurses for a shift
must be at least the shift’s demand.

```python
class DemandSatisfactionModule(ShiftAssignmentModule):
    """
    Ensure each shift meets its demand.
    """
    def build(
        self,
        instance: NurseRosteringInstance,
        model: cp_model.CpModel,
        nurse_shift_vars: list[NurseDecisionVars],
    ) -> cp_model.LinearExprT:
        for shift in instance.shifts:
            assigned_nurses = [
                nurse.is_assigned_to(shift_uid=shift.uid)
                for nurse in nurse_shift_vars
                if shift.uid in nurse._x
            ]
            model.add(sum(assigned_nurses) >= shift.demand)
        return 0
```

### Prefer Staff

While the previous modules defined hard constraints that must be satisfied, this
module introduces a **soft constraint**. We prefer to assign internal staff
nurses over contractors, as contractors may be more expensive or less desirable
for other reasons. Instead of enforcing this rule strictly, we **penalize** the
use of contractors in the objective function. This way, the solver will try to
minimize the number of contractor assignments while still satisfying all hard
constraints.

**Example Test:**

```python
from nurserostering.modules import PreferStaffModule, DemandSatisfactionModule
from cpsat_utils.testing import assert_objective
from generate import create_shifts, create_nurse

from nurserostering.nurse_vars import NurseDecisionVars
from nurserostering.data_schema import NurseRosteringInstance
from ortools.sat.python import cp_model


def test_prefer_staff_module():
    """
    Prefer assigning the staff nurse (objective = 0).
    """
    shifts = create_shifts(1)
    shifts[0].demand = 1

    staff = create_nurse("Staff", staff=True)
    contractor = create_nurse("Contractor", staff=False)
    instance = NurseRosteringInstance(nurses=[staff, contractor], shifts=shifts)

    model = cp_model.CpModel()
    vars_staff = NurseDecisionVars(staff, shifts, model)
    vars_contractor = NurseDecisionVars(contractor, shifts, model)
    solver = cp_model.CpSolver()
    staff_mod = PreferStaffModule()

    DemandSatisfactionModule().build(instance, model, [vars_staff, vars_contractor])
    model.minimize(staff_mod.build(instance, model, [vars_staff, vars_contractor]))

    assert_objective(model=model, solver=solver, expected=0.0)
    assert solver.value(vars_staff.is_assigned_to(shifts[0].uid)) == 1
    assert solver.value(vars_contractor.is_assigned_to(shifts[0].uid)) == 0
```

**Implementation:**

```python
class PreferStaffModule(ShiftAssignmentModule):
    def build(
        self,
        instance: NurseRosteringInstance,
        model: cp_model.CpModel,
        nurse_shift_vars: list[NurseDecisionVars],
    ) -> cp_model.LinearExprT:
        """
        Penalize use of non-staff (contract) nurses in the objective.
        """
        expr = 0
        for nv in nurse_shift_vars:
            if not nv.nurse.staff:
                for uid in nv._x:
                    expr += instance.staff_weight * nv.is_assigned_to(uid)
        return expr
```

### Preferences Objective

While many of the previous modules focused on enforcing constraints,
`MaximizePreferences` is different: it defines what we actually want to
optimize. Nurses often have preferences for specific shifts—perhaps due to
personal schedules, skill matching, or desired working hours. A good schedule
should take these preferences into account as much as possible.

The `MaximizePreferences` module encodes this by assigning a **negative
contribution** (a reward) to the objective function whenever a nurse is assigned
to one of their preferred shifts. As our interface will minimize by default, we
negate the weight so that preferred assignments reduce the total objective
value. Note that we could have easily specified that the expression returned by
`build` should be maximized.

This module is therefore not a soft constraint but a **true objective**: among
all feasible schedules, it guides the solver to find one that maximizes nurse
satisfaction.

**Example Test:**

```python
from nurserostering.modules import MaximizePreferences, DemandSatisfactionModule
from cpsat_utils.testing import assert_objective
from generate import create_shifts, create_nurse

from nurserostering.nurse_vars import NurseDecisionVars
from nurserostering.data_schema import NurseRosteringInstance

from ortools.sat.python import cp_model


def test_maximize_preferences_module():
    """
    Prefer assigning the nurse to their preferred shift (objective = -1).
    """
    shifts = create_shifts(1)
    shifts[0].demand = 1

    nurse = create_nurse("Preferred Nurse", preferred_shifts={shifts[0].uid})
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)

    model = cp_model.CpModel()
    nurse_vars = NurseDecisionVars(nurse, shifts, model)
    solver = cp_model.CpSolver()
    pref_mod = MaximizePreferences()

    DemandSatisfactionModule().build(instance, model, [nurse_vars])
    model.minimize(pref_mod.build(instance, model, [nurse_vars]))

    assert_objective(model=model, solver=solver, expected=-1.0)
    assert (
        solver.value(nurse_vars.is_assigned_to(shifts[0].uid)) == 1
    ), "Nurse should be assigned to their preferred shift"
```

**Implementation:**

```python
class MaximizePreferences(ShiftAssignmentModule):
    def build(
        self,
        instance: NurseRosteringInstance,
        model: cp_model.CpModel,
        nurse_shift_vars: list[NurseDecisionVars],
    ) -> cp_model.LinearExprT:
        """
        Reward assignments that match nurse preferences by reducing the objective value.
        """
        expr = 0
        for nv in nurse_shift_vars:
            for uid in nv.nurse.preferred_shifts:
                expr += -nv.nurse.preferred_shift_weight * nv.is_assigned_to(uid)
        return expr
```

## Full Solver

<!-- Introduces the solver class as the orchestration of all modules -->

At this point, we have implemented all the necessary modules: the constraints,
soft constraints, and objectives. The next step is to combine these components
into a complete optimization model. The `NurseRosteringModel` class is
responsible for orchestrating all modules, building the CP-SAT model, and
extracting the solution.

```python
from ortools.sat.python import cp_model
from .nurse_vars import NurseDecisionVars
from .data_schema import NurseRosteringInstance, NurseRosteringSolution
from .modules import (
    ShiftAssignmentModule,
    NoBlockedShiftsModule,
    DemandSatisfactionModule,
    MinTimeBetweenShifts,
    MaximizePreferences,
    PreferStaffModule,
)


class NurseRosteringModel:
    """
    A compact and extensible solver for the nurse rostering problem using CP-SAT.
    """

    def __init__(
        self, instance: NurseRosteringInstance, model: cp_model.CpModel | None = None
    ):
        self.instance = instance
        self.model = model or cp_model.CpModel()
        self.nurse_vars = [
            NurseDecisionVars(nurse, instance.shifts, self.model)
            for nurse in instance.nurses
        ]

        self.modules: list[ShiftAssignmentModule] = [
            NoBlockedShiftsModule(),
            DemandSatisfactionModule(),
            MinTimeBetweenShifts(),
            MaximizePreferences(),
            PreferStaffModule(),
        ]

        objective = sum(
            module.build(instance, self.model, self.nurse_vars)  # type: ignore
            for module in self.modules
        )
        self.model.minimize(objective)

    def solve(
        self,
        log_search_progress: bool = True,
        max_time_in_seconds: float = 60.0,
        **solver_params,
    ) -> NurseRosteringSolution:
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = log_search_progress
        solver.parameters.max_time_in_seconds = max_time_in_seconds
        for key, value in solver_params.items():
            setattr(solver.parameters, key, value)

        status = solver.solve(self.model)
        if status == cp_model.INFEASIBLE:
            raise ValueError("The model is infeasible.")
        elif status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError("Solver failed to find a feasible solution.")

        nurses_at_shifts = {}
        for nurse_model in self.nurse_vars:
            for shift_uid in nurse_model.extract(solver):
                nurses_at_shifts.setdefault(shift_uid, []).append(nurse_model.nurse.uid)

        return NurseRosteringSolution(
            nurses_at_shifts=nurses_at_shifts,
            objective_value=round(solver.objective_value),
        )
```

<!-- Introduces the example test instance -->

To demonstrate the full workflow, we can test the solver with a small but
non-trivial instance. The following test sets up 7 days with 3 shifts per day
(morning, day, and night), resulting in 21 shifts. The nurses have a mix of
preferences, blocked shifts, and max shift limits, creating an interesting
scheduling challenge.

```python
def test_fixed_instance():
    base_date = datetime(2025, 1, 1)
    shift_length = 8
    shifts = []

    # 7 days, 3 shifts/day = 21 shifts
    for day in range(7):
        for idx, hour in enumerate([0, 8, 16]):  # Morning, Day, Night
            start = base_date + timedelta(days=day, hours=hour)
            end = start + timedelta(hours=shift_length)
            shifts.append(
                Shift(
                    name=f"Day {day+1} Shift {idx+1}",
                    start_time=start,
                    end_time=end,
                    demand=2 if hour == 8 else 1,  # Higher demand for day shifts
                )
            )

    # Total demand: 7*2 (day) + 7*1 (morning) + 7*1 (night) = 28 + 7 + 7 = 42 shifts needed

    nurses = [
        Nurse(
            name="Alice (prefers mornings, no Sundays)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 0},
            blocked_shifts={s.uid for s in shifts if s.start_time.weekday() == 6},
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=3,
        ),
        Nurse(
            name="Bob (prefers nights)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 16},
            blocked_shifts=set(),
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=3,
        ),
        Nurse(
            name="Clara (prefers weekends, blocks weekdays)",
            preferred_shifts={
                s.uid for s in shifts if s.start_time.weekday() in {5, 6}
            },
            blocked_shifts={s.uid for s in shifts if s.start_time.weekday() < 5},
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=2,
        ),
        Nurse(
            name="Dan (no preferences)",
            preferred_shifts=set(),
            blocked_shifts=set(),
            staff=True,
            min_time_between_shifts=timedelta(hours=8),
            preferred_shift_weight=1,
        ),
        Nurse(
            name="Eve (contractor, prefers day shifts)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 8},
            blocked_shifts=set(),
            staff=False,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=2,
        ),
        Nurse(
            name="Frank (prefers day shifts)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 8},
            blocked_shifts=set(),
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=2,
        ),
        Nurse(
            name="Grace (prefers mornings)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 0},
            blocked_shifts=set(),
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=3,
        ),
        Nurse(
            name="Heidi (contractor, no preferences)",
            preferred_shifts=set(),
            blocked_shifts=set(),
            staff=False,
            min_time_between_shifts=timedelta(hours=8),
            preferred_shift_weight=1,
        ),
    ]

    instance = NurseRosteringInstance(
        nurses=nurses, shifts=sorted(shifts, key=lambda s: s.start_time), staff_weight=2
    )

    model = NurseRosteringModel(instance)
    solution = model.solve()
    assert solution is not None, "The solution should not be None"
    assert_solution_is_feasible(instance, solution)
```

#### Nurses

| Name & Description                            | Staff/Contractor | Min Rest | Preferences         | Blocked Shifts |
| --------------------------------------------- | ---------------- | -------- | ------------------- | -------------- |
| **Alice** (prefers mornings, no Sundays)      | Staff            | ≥ 10h    | 7 shifts (weight 3) | 3 shifts       |
| **Bob** (prefers nights)                      | Staff            | ≥ 10h    | 7 shifts (weight 3) | —              |
| **Clara** (prefers weekends, blocks weekdays) | Staff            | ≥ 10h    | 6 shifts (weight 2) | 15 shifts      |
| **Dan** (no preferences)                      | Staff            | ≥ 8h     | —                   | —              |
| **Eve** (contractor, prefers day shifts)      | Contractor       | ≥ 10h    | 7 shifts (weight 2) | —              |
| **Frank** (prefers day shifts)                | Staff            | ≥ 10h    | 7 shifts (weight 2) | —              |
| **Grace** (prefers mornings)                  | Staff            | ≥ 10h    | 7 shifts (weight 3) | —              |
| **Heidi** (contractor, no preferences)        | Contractor       | ≥ 8h     | —                   | —              |

#### Shift Coverage

| Day | Morning (S1)        | Day (S2)   | Night (S3) |
| --- | ------------------- | ---------- | ---------- |
| D1  | Alice, Grace        | Dan, Frank | Bob        |
| D2  | Alice, Grace        | Dan, Frank | Bob        |
| D3  | Alice, Grace        | Eve, Frank | Bob        |
| D4  | Alice, Clara, Grace | Dan, Frank | Bob        |
| D5  | Clara, Grace        | Eve, Frank | Bob        |
| D6  | Alice, Grace        | Eve, Frank | Bob        |
| D7  | Alice, Grace        | Dan, Frank | Bob        |

## Automatic Test Extraction from Production

Because we use pydantic for input and output, serializing test cases from
production is straightforward. You can save instances and solutions to JSON and
re-load them later for regression tests.

```python
# export instance and solution to JSON
with open("instance.json", "w") as f:
    f.write(instance.model_dump_json())
with open("solution.json", "w") as f:
    f.write(solution.model_dump_json())
```

You can then load these files in your tests:

```python
from nurserostering.data_schema import NurseRosteringInstance, NurseRosteringSolution

def test_load_instance_and_solution():
    with open("instance.json", "r") as f:
        instance = NurseRosteringInstance.model_validate_json(f.read())
    with open("solution.json", "r") as f:
        solution = NurseRosteringSolution.model_validate_json(f.read())

    assert instance is not None, "Instance should not be None"
    assert solution is not None, "Solution should not be None"
    # should still be feasible
    assert_solution_is_feasible(instance, solution)

    # reoptimize the solution
    model = NurseRosteringModel(instance)
    new_solution = model.solve(max_time_in_seconds=10.0)
    assert new_solution is not None, "New solution should not be None"
    assert_solution_is_feasible(instance, new_solution)
    assert new_solution.objective_value == solution.objective_value, "Objective value should match"
```

Actually, you can also just dump them into a folder and then make the test
iterate over all the files in that folder. Instead of just writing the objective
value into the solution, you could also add the time it took to solve the
instance, which can be useful for performance testing. With a bunch of such
files collected, you could then start to refactor and optimize your code, with a
minimum of regression risk in correctness or performance. However, keep in mind
that the performance of CP-SAT (and other such solvers) can vary between runs,
so do not be too strict with the performance tests.

Instead of relying solely on historical data for testing, you can **shadow**
your production model with a development or experimental model. This approach
allows you not only to verify correctness but also to directly compare the
performance of both models, providing strong confidence that your modifications
enhance the model without disrupting production. If you already have a
monitoring tool in place, integrating _shadow solutions_ is likely
straightforward. This technique is so popular that
[nextmv](https://www.nextmv.io/docs/using-nextmv/experiments/shadow) offers
built-in support for it.

## Conclusion

In this chapter, we built a complete nurse rostering solver using CP-SAT and a
test-driven approach. The modular design allowed us to separate constraints and
objectives, write targeted tests, and iterate safely. This methodology is not
always the fastest way to get a working solution, but it pays off for problems
that evolve over time.

> [!TIP]
>
> This chapter primarily emphasized correctness. However, once your solution is
> working correctly, you may discover that it does not scale sufficiently well.
> In the chapter [Benchmarking your Model](#08-benchmarking), you will learn how
> to benchmark models to evaluate improvements in performance and scalability.
> The testing framework developed here helps maintain correctness during
> performance optimization. Frequently, enhancing efficiency involves slight
> modifications to the problem formulation—such as approximating non-linear
> elements with linear ones or relaxing constraints that can usually be repaired
> in postprocessing with minimal loss of solution quality. Adapting these
> changes may require adjustments to your tests, but typically you can reuse
> many of them.
