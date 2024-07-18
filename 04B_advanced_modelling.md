<a name="04B-advanced-modelling"></a>

## Advanced Modeling

After having seen the basic elements of CP-SAT, this chapter will introduce you
to the more complex constraints. These constraints are already focussed on
specific problems, such as routing or scheduling, but very generic and powerful
within their domain. However, they also need more explanation on the correct
usage.

- [Tour Constraints](#04-modelling-circuit): `add_circuit`,
  `add_multiple_circuit`
- [Automaton Constraints](#04-modelling-automaton): `add_automaton`
- [Reservoir Constraints](#04-modelling-reservoir): `add_reservoir_constraint`,
  `add_reservoir_constraint_with_active`
- [Intervals](#04-modelling-intervals): `new_interval_var`,
  `new_interval_var_series`, `new_fixed_size_interval_var`,
  `new_optional_interval_var`, `new_optional_interval_var_series`,
  `new_optional_fixed_size_interval_var`,
  `new_optional_fixed_size_interval_var_series`,
  `add_no_overlap`,`add_no_overlap_2d`, `add_cumulative`
- [Piecewise Linear Constraints](#04-modelling-pwl): Not officially part of
  CP-SAT, but we provide some free copy&pasted code to do it.

<a name="04-modelling-circuit"></a>

### Circuit/Tour-Constraints

Routes and tours are essential in addressing optimization challenges across
various fields, far beyond traditional routing issues. For example, in DNA
sequencing, optimizing the sequence in which DNA fragments are assembled is
crucial, while in scientific research, methodically ordering the reconfiguration
of experiments can greatly reduce operational costs and downtime. The
`add_circuit` and `add_multiple_circuit` constraints in CP-SAT allow you to
easily model various scenarios. These constraints extend beyond the classical
[Traveling Salesman Problem (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem),
allowing for solutions where not every node needs to be visited and
accommodating scenarios that require multiple disjoint sub-tours. This
adaptability makes them invaluable for a broad spectrum of practical problems
where the sequence and arrangement of operations critically impact efficiency
and outcomes.

|                         ![TSP Example](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/optimal_tsp.png)                         |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: |
| The Traveling Salesman Problem (TSP) asks for the shortest possible route that visits every vertex exactly once and returns to the starting vertex. |

The Traveling Salesman Problem is one of the most famous and well-studied
combinatorial optimization problems. It is a classic example of a problem that
is easy to understand, common in practice, but hard to solve. It also has a
special place in the history of optimization, as many techniques that are now
used generally were first developed for the TSP. If you have not done so yet, I
recommend watching
[this talk by Bill Cook](https://www.youtube.com/watch?v=5VjphFYQKj8), or even
reading the book
[In Pursuit of the Traveling Salesman](https://press.princeton.edu/books/paperback/9780691163529/in-pursuit-of-the-traveling-salesman).

> [!TIP]
>
> If your problem is specifically the Traveling Salesperson Problem (TSP), you
> might find the
> [Concorde solver](https://www.math.uwaterloo.ca/tsp/concorde.html)
> particularly effective. For problems closely related to the TSP, a Mixed
> Integer Programming (MIP) solver may be more suitable, as many TSP variants
> yield strong linear programming relaxations that MIP solvers can efficiently
> exploit. Additionally, consider
> [OR-Tools Routing](https://developers.google.com/optimization/routing) if
> routing constitutes a significant aspect of your problem. However, for
> scenarios where variants of the TSP are merely a component of a larger
> problem, utilizing CP-SAT with the `add_circuit` or `add_multiple_circuit`
> constraints can be very beneficial.

|                                                                                                                                                        ![TSP BnB Example](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/tsp_bnb_improved.png)                                                                                                                                                         |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| This example shows why Mixed Integer Programming solvers are so good in solving the TSP. The linear relaxation (at the top) is already very close to the optimal solution. By branching, i.e., trying 0 and 1, on just two fractional variables, we not only find the optimal solution but can also prove optimality. The example was generated with the [DIY TSP Solver](https://www.math.uwaterloo.ca/tsp/D3/bootQ.html). |

#### `add_circuit`

The `add_circuit` constraint is utilized to solve circuit problems within
directed graphs, even allowing loops. It operates by taking a list of triples
`(u,v,var)`, where `u` and `v` denote the source and target vertices,
respectively, and `var` is a Boolean variable that indicates if an edge is
included in the solution. The constraint ensures that the edges marked as `True`
form a single circuit visiting each vertex exactly once, aside from vertices
with a loop set as `True`. Vertex indices should start at 0 and must not be
skipped to avoid isolation and infeasibility in the circuit.

Here is an example using the CP-SAT solver to address a directed Traveling
Salesperson Problem (TSP):

```python
from ortools.sat.python import cp_model

# Directed graph with weighted edges
dgraph = {(0, 1): 13, (1, 0): 17, ...(2, 3): 27}

# Initialize CP-SAT model
model = cp_model.CpModel()

# Boolean variables for each edge
edge_vars = {(u, v): model.new_bool_var(f"e_{u}_{v}") for (u, v) in dgraph.keys()}

# Circuit constraint for a single tour
model.add_circuit([(u, v, var) for (u, v), var in edge_vars.items()])

# Objective function to minimize total cost
model.minimize(sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items()))

# Solve model
solver = cp_model.CpSolver()
status = solver.solve(model)
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    tour = [(u, v) for (u, v), x in edge_vars.items() if solver.value(x)]
    print("Tour:", tour)

# Output: [(0, 1), (2, 0), (3, 2), (1, 3)], i.e., 0 -> 1 -> 3 -> 2 -> 0
```

This constraint can be adapted for paths by adding a virtual enforced edge that
closes the path into a circuit, such as `(3, 0, 1)` for a path from vertex 0 to
vertex 3.

#### Creative usage of `add_circuit`

The `add_circuit` constraint can be creatively adapted to solve various related
problems. While there are more efficient algorithms for solving the Shortest
Path Problem, let us demonstrate how to adapt the `add_circuit` constraint for
educational purposes.

```python
from ortools.sat.python import cp_model

# Define a weighted, directed graph with edge costs
dgraph = {(0, 1): 13, (1, 0): 17, ...(2, 3): 27}

source_vertex = 0
target_vertex = 3

# Add zero-cost loops for vertices not being the source or target
for v in [1, 2]:
    dgraph[(v, v)] = 0

# Initialize CP-SAT model and variables
model = cp_model.CpModel()
edge_vars = {(u, v): model.new_bool_var(f"e_{u}_{v}") for (u, v) in dgraph}

# Define the circuit including a pseudo-edge from target to source
circuit = [(u, v, var) for (u, v), var in edge_vars.items()] + [
    (target_vertex, source_vertex, 1)
]
model.add_circuit(circuit)

# Minimize total cost
model.minimize(sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items()))

# Solve and extract the path
solver = cp_model.CpSolver()
status = solver.solve(model)
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    path = [(u, v) for (u, v), x in edge_vars.items() if solver.value(x) and u != v]
    print("Path:", path)

# Output: [(0, 1), (1, 3)], i.e., 0 -> 1 -> 3
```

This approach showcases the flexibility of the `add_circuit` constraint for
various tour and path problems. Explore further examples:

- [Budget constrained tours](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_circuit_budget.py):
  Optimize the largest possible tour within a specified budget.
- [Multiple tours](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_circuit_multi_tour.py):
  Solve for $k$ minimal tours covering all vertices.

#### `add_multiple_circuit`

You can model multiple disjoint tours using several `add_circuit` constraints,
as demonstrated in
[this example](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_circuit_multi_tour.py).
If all tours share a common depot (vertex 0), the `add_multiple_circuit`
constraint is an alternative. However, this constraint does not allow you to
specify the number of tours, nor can it determine to which tour a particular
edge belongs. Therefore, the `add_circuit` constraint is often a superior
choice. Although the arguments for both constraints are identical, vertex 0
serves a unique role as the depot where all tours commence and conclude.

#### Performance of `add_circuit` for the TSP

The table below displays the performance of the CP-SAT solver on various
instances of the TSPLIB, using the `add_circuit` constraint, under a 90-second
time limit. The performance can be considered reasonable, but can be easily
beaten by a Mixed Integer Programming solver.

| Instance | # vertices | runtime | lower bound | objective | opt. gap |
| :------- | ---------: | ------: | ----------: | --------: | -------: |
| att48    |         48 |    0.47 |       33522 |     33522 |        0 |
| eil51    |         51 |    0.69 |         426 |       426 |        0 |
| st70     |         70 |     0.8 |         675 |       675 |        0 |
| eil76    |         76 |    2.49 |         538 |       538 |        0 |
| pr76     |         76 |   54.36 |      108159 |    108159 |        0 |
| kroD100  |        100 |    9.72 |       21294 |     21294 |        0 |
| kroC100  |        100 |    5.57 |       20749 |     20749 |        0 |
| kroB100  |        100 |     6.2 |       22141 |     22141 |        0 |
| kroE100  |        100 |    9.06 |       22049 |     22068 |        0 |
| kroA100  |        100 |    8.41 |       21282 |     21282 |        0 |
| eil101   |        101 |    2.24 |         629 |       629 |        0 |
| lin105   |        105 |    1.37 |       14379 |     14379 |        0 |
| pr107    |        107 |     1.2 |       44303 |     44303 |        0 |
| pr124    |        124 |    33.8 |       59009 |     59030 |        0 |
| pr136    |        136 |   35.98 |       96767 |     96861 |        0 |
| pr144    |        144 |   21.27 |       58534 |     58571 |        0 |
| kroB150  |        150 |   58.44 |       26130 |     26130 |        0 |
| kroA150  |        150 |   90.94 |       26498 |     26977 |       2% |
| pr152    |        152 |   15.28 |       73682 |     73682 |        0 |
| kroA200  |        200 |   90.99 |       29209 |     29459 |       1% |
| kroB200  |        200 |   31.69 |       29437 |     29437 |        0 |
| pr226    |        226 |   74.61 |       80369 |     80369 |        0 |
| gil262   |        262 |   91.58 |        2365 |      2416 |       2% |
| pr264    |        264 |   92.03 |       49121 |     49512 |       1% |
| pr299    |        299 |   92.18 |       47709 |     49217 |       3% |
| linhp318 |        318 |   92.45 |       41915 |     52032 |      19% |
| lin318   |        318 |   92.43 |       41915 |     52025 |      19% |
| pr439    |        439 |   94.22 |      105610 |    163452 |      35% |

There are two prominent formulations to model the Traveling Salesman Problem
(TSP) without an `add_circuit` constraint: the
[Dantzig-Fulkerson-Johnson (DFJ) formulation](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Dantzig%E2%80%93Fulkerson%E2%80%93Johnson_formulation)
and the
[Miller-Tucker-Zemlin (MTZ) formulation](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation[21]).
The DFJ formulation is generally regarded as more efficient due to its stronger
linear relaxation. However, it requires lazy constraints, which are not
supported by the CP-SAT solver. When implemented without lazy constraints, the
performance of the DFJ formulation is comparable to that of the MTZ formulation
in CP-SAT. Nevertheless, both formulations perform significantly worse than the
`add_circuit` constraint. This indicates the superiority of using the
`add_circuit` constraint for handling tours and paths in such problems. Unlike
end users, the `add_circuit` constraint can utilize lazy constraints internally,
offering a substantial advantage in solving the TSP.

<a name="04-modelling-intervals"></a>

### Scheduling and Packing with Intervals

A special case of variables are the interval variables, that allow to model
intervals, i.e., a span of some length with a start and an end. There are fixed
length intervals, flexible length intervals, and optional intervals to model
various use cases. These intervals become interesting in combination with the
no-overlap constraints for 1D and 2D. We can use this for geometric packing
problems, scheduling problems, and many other problems, where we have to prevent
overlaps between intervals. These variables are special because they are
actually not a variable, but a container that bounds separately defined start,
length, and end variables.

There are four types of interval variables: `new_interval_var`,
`new_fixed_size_interval_var`, `new_optional_interval_var`, and
`new_optional_fixed_size_interval_var`. The `new_optional_interval_var` is the
most expressive but also the most expensive, while `new_fixed_size_interval_var`
is the least expressive and the easiest to optimize. All four types take a
`start=` variable. Intervals with `fixed_size` in their name require a constant
`size=` argument defining the interval length. Otherwise, the `size=` argument
can be a variable in combination with an `end=` variable, which complicates the
solution. Intervals with `optional` in their name include an `is_present=`
argument, a boolean indicating if the interval is present. The no-overlap
constraints, discussed later, apply only to intervals that are present, allowing
for modeling problems with multiple resources or optional tasks. Instead of a
pure integer variable, all arguments also accept an affine expression, e.g.,
`start=5*start_var+3`.

```python
model = cp_model.CpModel()

start_var = model.new_int_var(0, 100, "start")
length_var = model.new_int_var(10, 20, "length")
end_var = model.new_int_var(0, 100, "end")
is_present_var = model.new_bool_var("is_present")

# creating an interval whose length can be influenced by a variable (more expensive)
flexible_interval = model.new_interval_var(
    start=start_var, size=length_var, end=end_var, name="flexible_interval"
)

# creating an interval of fixed length
fixed_interval = model.new_fixed_size_interval_var(
    start=start_var,
    size=10,  # needs to be a constant
    name="fixed_interval",
)

# creating an interval that can be present or not and whose length can be influenced by a variable (most expensive)
optional_interval = model.new_optional_interval_var(
    start=start_var,
    size=length_var,
    end=end_var,
    is_present=is_present_var,
    name="optional_interval",
)

# creating an interval that can be present or not
optional_fixed_interval = model.new_optional_fixed_size_interval_var(
    start=start_var,
    size=10,  # needs to be a constant
    is_present=is_present_var,
    name="optional_fixed_interval",
)
```

These interval variables are not useful on their own, as we could have easily
achieved the same with a simple linear constraint. However, CP-SAT provides
special constraints for these interval variables, that would actually be much
harder to model by hand and are also much more efficient.

CP-SAT offers the following three constraints for intervals:
`add_no_overlap`,`add_no_overlap_2d`, `add_cumulative`. `add_no_overlap` is used
to prevent overlaps between intervals on a single dimension, e.g., time.
`add_no_overlap_2d` is used to prevent overlaps between intervals on two
dimensions, e.g., for packing rectangles. `add_cumulative` is used to model a
resource constraint, where the sum of the demands of the overlapping intervals
must not exceed the capacity of the resource.

```python
optional_interval = model.new_optional_interval_var(
    start=start_var,
    size=length_var,
    end=end_var,
    is_present=is_present_var,
    name="optional_interval",
)

# creating an interval that can be present or not
optional_fixed_interval = model.new_optional_fixed_size_interval_var(
    start=start_var,
    size=10,  # needs to be a constant
    is_present=is_present_var,
    name="optional_fixed_interval",
)

model.add_no_overlap(
    interval_vars=[
        flexible_interval,
        fixed_interval,
        optional_interval,
        optional_fixed_interval,
    ]
)
model.add_no_overlap_2d(
    x_intervals=[
        flexible_interval,
        fixed_interval,
        optional_interval,
        optional_fixed_interval,
    ],
    y_intervals=[
        flexible_interval,
        fixed_interval,
        optional_interval,
        optional_fixed_interval,
    ],
)

demand_vars = [model.new_int_var(1, 10, f"demand_{i}") for i in range(4)]
capacity_var = model.new_int_var(1, 100, "capacity")
model.add_cumulative(
    intervals=[
        flexible_interval,
        fixed_interval,
        optional_interval,
        optional_fixed_interval,
    ],
    demands=demand_vars,
    capacity=capacity_var,
)
```

Let us examine a few examples of how to use these constraints effectively.

#### Scheduling for a Conference Room with Intervals

Assume we have a conference room and need to schedule several meetings. Each
meeting has a fixed length and a range of possible start times. The time slots
are in 5-minute intervals starting at 8:00 AM and ending at 6:00 PM. Thus, there
are $10 \times 12 = 120$ time slots, and we can use a simple integer variable to
model the start time. With fixed meeting lengths, we can use the
`new_fixed_size_interval_var` to model the intervals. The `add_no_overlap`
constraint ensures no two meetings overlap, and domains for the start time can
model the range of possible start times.

```python
# Convert time to index and back
def t_to_idx(hour, minute):
    return (hour - 8) * 12 + minute // 5


def idx_to_t(timepoint):
    hour = 8 + timepoint // 12
    minute = (timepoint % 12) * 5
    return f"{hour}:{minute:02d}"


# Define meeting information using namedtuples
MeetingInfo = namedtuple("MeetingInfo", ["start_times", "duration"])

# Meeting definitions
meetings = {
    "meeting_a": MeetingInfo(
        start_times=[
            [t_to_idx(8, 0), t_to_idx(12, 0)],
            [t_to_idx(16, 0), t_to_idx(17, 0)],
        ],
        duration=24,  # 2 hours
    ),
    "meeting_b": MeetingInfo(
        start_times=[
            [t_to_idx(10, 0), t_to_idx(12, 0)],
        ],
        duration=6,  # 30 minutes
    ),
    "meeting_c": MeetingInfo(
        start_times=[
            [t_to_idx(16, 0), t_to_idx(17, 0)],
        ],
        duration=3,  # 15 minutes
    ),
    "meeting_d": MeetingInfo(
        start_times=[
            [t_to_idx(8, 0), t_to_idx(10, 0)],
            [t_to_idx(12, 0), t_to_idx(14, 0)],
        ],
        duration=12,  # 1 hour
    ),
}

# Create a new CP-SAT model
model = cp_model.CpModel()

# Create start time variables for each meeting
start_time_vars = {
    meeting_name: model.new_int_var_from_domain(
        cp_model.Domain.from_intervals(meeting_info.start_times),
        f"start_{meeting_name}",
    )
    for meeting_name, meeting_info in meetings.items()
}

# Create interval variables for each meeting
interval_vars = {
    meeting_name: model.new_fixed_size_interval_var(
        start=start_time_vars[meeting_name],
        size=meeting_info.duration,
        name=f"interval_{meeting_name}",
    )
    for meeting_name, meeting_info in meetings.items()
}

# Add the no-overlap constraint to the model
model.add_no_overlap(list(interval_vars.values()))

# Solve the model
solver = cp_model.CpSolver()
status = solver.solve(model)

# Extract and print the solution
scheduled_times = {}
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for meeting_name in meetings:
        start_time = solver.value(start_time_vars[meeting_name])
        scheduled_times[meeting_name] = start_time
        print(f"{meeting_name} starts at {idx_to_t(start_time)}")
else:
    print("No feasible solution found.")
```

This will result in a solution like this:

|                ![Schedule](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/scheduling_example.png)                |
| :-----------------------------------------------------------------------------------------------------------------------------------: |
| A possible non-overlapping schedule for the above example. The instance is quite simple, but you could try adding some more meetings. |

#### Scheduling for Multiple Resources with Optional Intervals

Now, imagine we have multiple resources, such as multiple conference rooms, and
we need to schedule the meetings such that no two meetings overlap in the same
room. This can be modeled with optional intervals, where the intervals exist
only if the meeting is scheduled in the room. The `add_no_overlap` constraint
ensures that no two meetings overlap in the same room.

```python
# Meeting definitions
meetings = {
    "meeting_a": MeetingInfo(
        start_times=[
            [t_to_idx(8, 0), t_to_idx(12, 0)],
            [t_to_idx(16, 0), t_to_idx(16, 0)],
        ],
        duration=24,
    ),
    "meeting_b": MeetingInfo(
        start_times=[[t_to_idx(10, 0), t_to_idx(12, 0)]], duration=48
    ),
    "meeting_c": MeetingInfo(
        start_times=[[t_to_idx(16, 0), t_to_idx(17, 0)]], duration=6
    ),
    "meeting_d": MeetingInfo(
        start_times=[
            [t_to_idx(8, 0), t_to_idx(10, 0)],
            [t_to_idx(12, 0), t_to_idx(14, 0)],
        ],
        duration=12,
    ),
    "meeting_e": MeetingInfo(
        start_times=[[t_to_idx(10, 0), t_to_idx(12, 0)]], duration=24
    ),
    "meeting_f": MeetingInfo(
        start_times=[[t_to_idx(14, 0), t_to_idx(14, 0)]], duration=48
    ),
    "meeting_g": MeetingInfo(
        start_times=[[t_to_idx(14, 0), t_to_idx(16, 0)]], duration=24
    ),
}

# Create the model
model = cp_model.CpModel()

# Create start time and room variables
start_time_vars = {
    name: model.new_int_var_from_domain(
        cp_model.Domain.from_intervals(info.start_times), f"start_{name}"
    )
    for name, info in meetings.items()
}

rooms = ["room_a", "room_b"]
room_vars = {
    name: {room: model.new_bool_var(f"{name}_in_{room}") for room in rooms}
    for name in meetings
}

# Ensure each meeting is assigned to exactly one room
for name, room_dict in room_vars.items():
    model.add_exactly_one(room_dict.values())

# Create interval variables and add no-overlap constraint
interval_vars = {
    name: {
        # We need a separate interval for each room
        room: model.new_optional_fixed_size_interval_var(
            start=start_time_vars[name],
            size=info.duration,
            is_present=room_vars[name][room],
            name=f"interval_{name}_in_{room}",
        )
        for room in rooms
    }
    for name, info in meetings.items()
}

for room in rooms:
    model.add_no_overlap([interval_vars[name][room] for name in meetings])
```

This will result in a solution like this:

| ![Schedule multiple rooms](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/scheduling_multiple_resources.png) |
| :-------------------------------------------------------------------------------------------------------------------------------: |
|                          A possible non-overlapping schedule for the above example with multiple rooms.                           |

> [!WARNING]
>
> Do not directly jump to intervals when you have a scheduling problem.
> Intervals are great if you actually have a somewhat continuous time or space
> that you need to schedule. If you have a more discrete problem, such as a
> scheduling problem with a fixed number of slots, you can often model this
> problem much more efficiently using simple Boolean variables and constraints.
> Especially if you can use domain knowledge to find clusters of meetings that
> cannot overlap, this can be much more efficient.

> [!TIP]
>
> You could easily extend this model to schedule as many meetings as feasible
> using an objective function. You could also maximize the distance between two
> meetings by using a variable size interval. This would be a good exercise to
> try.

#### Packing rectangles without overlaps

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
