<!-- EDIT THIS PART VIA 08_lns.md -->

## Using CP-SAT for Bigger Problems with Large Neighborhood Search
<a name="09-lns"></a>

CP-SAT is great at solving small and medium-sized problems. But what if you have
a really big problem on your hands? One option might be to use a special kind of
algorithm known as a "meta-heuristic", like a
[genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm). But these
can be hard to set up and might not even give you good results.

Sometimes you will see new algorithms with cool-sounding names in scientific
papers. While tempting, these are often just small twists on older methods and
might leave out key details that make them work. If you are interested, there's a
discussion about this issue in a paper by Sörensen, called
["Metaheuristics – The Metaphor Exposed"](http://onlinelibrary.wiley.com/doi/10.1111/itor.12001/).

The good news? You do not have to implement an algorithm that simulates the
mating behavior of forest frogs to solve your problem. If you already know how
to use CP-SAT, you can stick with it to solve big problems without adding
unnecessary complications. Even better? This technique, called Large
Neighborhood Search, often outperforms all other approaches.

### What Sets Large Neighborhood Search Apart?

Many traditional methods generate several "neighbor" options around an existing
solution and pick the best one. However, making each neighbor solution takes
time, limiting how many you can examine.

Large Neighborhood Search (LNS) offers a more efficient approach. Instead of
generating neighbors one by one, LNS formulates a "mini-problem" that modifies
parts of the current solution. This often involves randomly selecting some
variables, resetting them, and using CP-SAT (or a similar tool) to find the
optimal new values within the context of the remaining solution. This method,
known as "destroy and repair," allows for a broader exploration of neighbor
solutions without constructing each one individually.

Moreover, LNS can easily be mixed with other methods like genetic algorithms. If
you are already using a genetic algorithm, you could supercharge it by applying
CP-SAT to find the best possible crossover of two or more existing solutions. It
is like genetic engineering, but without any ethical worries!

When looking into the logs of CP-SAT, you may notice that it uses LNS itself to
find better solutions.

```
8 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns, graph_dec_lns, graph_var_lns, rins/rens, rnd_cst_lns, rnd_var_lns]
```

Why does it not suffice to just run CP-SAT if it already solves the problem with
LNS? The reason is that CP-SAT has to be relatively problem-agnostic. It has no
way of knowing the structure of your problem and thus cannot use this
information to improve the search. You on the other hand know a lot about your
problem and can use this knowledge to implement a more efficient version.

**Literature:**

- General Paper on LNS-variants:
  [Pisinger and Ropke - 2010](https://backend.orbit.dtu.dk/ws/portalfiles/portal/5293785/Pisinger.pdf)
- A generic variant (RINS), that is also used by CP-SAT:
  [Danna et al. 2005](https://link.springer.com/article/10.1007/s10107-004-0518-7)

We will now look into some examples to see this approach in action.

#### Example 1: Knapsack

You are given a knapsack that can carry a certain weight limit $C$, and you have
various items $I$ you can put into it. Each item $i\in I$ has a weight $w_i$ and
a value $v_i$. The goal is to pick items to maximize the total value while
staying within the weight limit.

$$\max \sum_{i \in I} v_i x_i$$

$$\text{s.t.} \sum_{i \in I} w_i x_i \leq C$$

$$x_i \in \\{0,1\\}$$

This is one of the simplest NP-hard problems and can be solved with a dynamic
programming approach in pseudo-polynomial time. CP-SAT is also able to solve
many large instances of this problem in an instant. However, its simple
structure makes it a good example to demonstrate the use of Large Neighborhood
Search, even if the algorithm will not be of much use for this problem.

A simple idea for the LNS is to delete some elements from the current solution,
compute the remaining capacity after deletion, select some additional items from
the remaining items, and try to find the optimal solution to fill the remaining
capacity with the deleted items and the newly selected items. Repeat this until
you are happy with the solution quality. The number of items you delete and
select can be fixed such that the problem can be easily solved by CP-SAT. You
can find a full implementation under
[examples/lns_knapsack.ipynb](examples/lns_knapsack.ipynb).

Let us look only on an example here:

Instance: $C=151$,
$I=I_{0}(w=12, v=37),I_{1}(w=16, v=49),I_{2}(w=20, v=53),I_{3}(w=11, v=14),I_{4}(w=19, v=42),$
$\quad I_{5}(w=13, v=53),I_{6}(w=18, v=54),I_{7}(w=16, v=56),I_{8}(w=14, v=45),I_{9}(w=12, v=39),$
$\quad I_{10}(w=11, v=42),I_{11}(w=19, v=43),I_{12}(w=12, v=43),I_{13}(w=19, v=66),I_{14}(w=20, v=54),$
$\quad I_{15}(w=13, v=54),I_{16}(w=12, v=33),I_{17}(w=12, v=38),I_{18}(w=14, v=43),I_{19}(w=15, v=28),$
$\quad I_{20}(w=11, v=47),I_{21}(w=10, v=31),I_{22}(w=20, v=97),I_{23}(w=10, v=35),I_{24}(w=19, v=56),$
$\quad I_{25}(w=11, v=33),I_{26}(w=12, v=38),I_{27}(w=15, v=45),I_{28}(w=17, v=58),I_{29}(w=11, v=48),$
$\quad I_{30}(w=15, v=32),I_{31}(w=17, v=67),I_{32}(w=15, v=43),I_{33}(w=16, v=41),I_{34}(w=18, v=42),$
$\quad I_{35}(w=14, v=44),I_{36}(w=20, v=45),I_{37}(w=13, v=50),I_{38}(w=17, v=57),I_{39}(w=17, v=33),$
$\quad I_{40}(w=17, v=49),I_{41}(w=12, v=21),I_{42}(w=14, v=37),I_{43}(w=20, v=74),I_{44}(w=14, v=55),$
$\quad I_{45}(w=10, v=25),I_{46}(w=16, v=26),I_{47}(w=10, v=37),I_{48}(w=18, v=63),I_{49}(w=16, v=39),$
$\quad I_{50}(w=16, v=57),I_{51}(w=16, v=47),I_{52}(w=10, v=43),I_{53}(w=12, v=30),I_{54}(w=12, v=40),$
$\quad I_{55}(w=19, v=48),I_{56}(w=12, v=39),I_{57}(w=14, v=43),I_{58}(w=17, v=35),I_{59}(w=19, v=51),$
$\quad I_{60}(w=16, v=48),I_{61}(w=19, v=72),I_{62}(w=16, v=45),I_{63}(w=19, v=88),I_{64}(w=15, v=20),$
$\quad I_{65}(w=17, v=49),I_{66}(w=14, v=40),I_{67}(w=14, v=27),I_{68}(w=19, v=51),I_{69}(w=10, v=37),$
$\quad I_{70}(w=15, v=42),I_{71}(w=13, v=29),I_{72}(w=20, v=87),I_{73}(w=13, v=28),I_{74}(w=15, v=38),$
$\quad I_{75}(w=19, v=77),I_{76}(w=13, v=35),I_{77}(w=17, v=55),I_{78}(w=13, v=39),I_{79}(w=10, v=26),$
$\quad I_{80}(w=15, v=32),I_{81}(w=12, v=40),I_{82}(w=11, v=21),I_{83}(w=18, v=82),I_{84}(w=13, v=41),$
$\quad I_{85}(w=12, v=27),I_{86}(w=15, v=35),I_{87}(w=18, v=48),I_{88}(w=15, v=64),I_{89}(w=19, v=62),$
$\quad I_{90}(w=20, v=64),I_{91}(w=13, v=45),I_{92}(w=19, v=64),I_{93}(w=18, v=83),I_{94}(w=11, v=38),$
$\quad I_{95}(w=10, v=30),I_{96}(w=18, v=65),I_{97}(w=19, v=56),I_{98}(w=12, v=41),I_{99}(w=17, v=36)$

Initial solution of value 442:
$\\{I_{0}, I_{1}, I_{2}, I_{3}, I_{4}, I_{5}, I_{6}, I_{7}, I_{8}, I_{9}\\}$

We will now repeatedly delete 5 items from the current solution and try to fill
the newly gained capacity with an optimal solution built from the deleted items
and 10 additional items. Note that this approach essentially considers
$2^{5+10}=32768$ neighbored solutions in each iteration. However, we could
easily scale it up to consider $2^{100+900}\sim 10^{300}$ neighbored solutions
in each iteration thanks to the implicit representation of the neighbored
solutions and CP-SAT ability to prune large parts of the search space.

**Round 1 of LNS algorithm:**

- Deleting the following 5 items from the solution:
  $\\{I_{0}, I_{7}, I_{8}, I_{9}, I_{6}\\}$
- Repairing solution by considering the following subproblem:
  - Subproblem: $C=72$,
    $I=\\{I_{6},I_{9},I_{86},I_{13},I_{47},I_{73},I_{0},I_{8},I_{7},I_{38},I_{57},I_{11},I_{60},I_{14}\\}$
- Computed the following solution of value 244 for the subproblem:
  $\\{I_{8}, I_{9}, I_{13}, I_{38}, I_{47}\\}$
- Combining
  $\\{I_{1}, I_{2}, I_{3}, I_{4}, I_{5}\\}\cup \\{I_{8}, I_{9}, I_{13}, I_{38}, I_{47}\\}$
- New solution of value 455:
  $\\{I_{1}, I_{2}, I_{3}, I_{4}, I_{5}, I_{8}, I_{9}, I_{13}, I_{38}, I_{47}\\}$

**Round 2 of LNS algorithm:**

- Deleting the following 5 items from the solution:
  $\\{I_{3}, I_{13}, I_{2}, I_{9}, I_{1}\\}$
- Repairing solution by considering the following subproblem:
  - Subproblem: $C=78$,
    $I=\\{I_{13},I_{9},I_{84},I_{41},I_{15},I_{42},I_{74},I_{16},I_{3},I_{1},I_{2},I_{67},I_{50},I_{89},I_{43}\\}$
- Computed the following solution of value 275 for the subproblem:
  $\\{I_{1}, I_{15}, I_{43}, I_{50}, I_{84}\\}$
- Combining
  $\\{I_{4}, I_{5}, I_{8}, I_{38}, I_{47}\\}\cup \\{I_{1}, I_{15}, I_{43}, I_{50}, I_{84}\\}$
- New solution of value 509:
  $\\{I_{1}, I_{4}, I_{5}, I_{8}, I_{15}, I_{38}, I_{43}, I_{47}, I_{50}, I_{84}\\}$

**Round 3 of LNS algorithm:**

- Deleting the following 5 items from the solution:
  $\\{I_{8}, I_{43}, I_{84}, I_{1}, I_{50}\\}$
- Repairing solution by considering the following subproblem:
  - Subproblem: $C=79$,
    $I=\\{I_{84},I_{76},I_{34},I_{16},I_{37},I_{20},I_{8},I_{43},I_{50},I_{1},I_{12},I_{35},I_{53}\\}$
- Computed the following solution of value 283 for the subproblem:
  $\\{I_{8}, I_{12}, I_{20}, I_{37}, I_{50}, I_{84}\\}$
- Combining
  $\\{I_{4}, I_{5}, I_{15}, I_{38}, I_{47}\\}\cup \\{I_{8}, I_{12}, I_{20}, I_{37}, I_{50}, I_{84}\\}$
- New solution of value 526:
  $\\{I_{4}, I_{5}, I_{8}, I_{12}, I_{15}, I_{20}, I_{37}, I_{38}, I_{47}, I_{50}, I_{84}\\}$

**Round 4 of LNS algorithm:**

- Deleting the following 5 items from the solution:
  $\\{I_{37}, I_{4}, I_{20}, I_{5}, I_{15}\\}$
- Repairing solution by considering the following subproblem:
  - Subproblem: $C=69$,
    $I=\\{I_{37},I_{4},I_{20},I_{15},I_{82},I_{41},I_{56},I_{76},I_{85},I_{5},I_{32},I_{57},I_{7},I_{67}\\}$
- Computed the following solution of value 260 for the subproblem:
  $\\{I_{5}, I_{7}, I_{15}, I_{20}, I_{37}\\}$
- Combining
  $\\{I_{8}, I_{12}, I_{38}, I_{47}, I_{50}, I_{84}\\}\cup \\{I_{5}, I_{7}, I_{15}, I_{20}, I_{37}\\}$
- New solution of value 540:
  $\\{I_{5}, I_{7}, I_{8}, I_{12}, I_{15}, I_{20}, I_{37}, I_{38}, I_{47}, I_{50}, I_{84}\\}$

**Round 5 of LNS algorithm:**

- Deleting the following 5 items from the solution:
  $\\{I_{38}, I_{12}, I_{20}, I_{47}, I_{37}\\}$
- Repairing solution by considering the following subproblem:
  - Subproblem: $C=66$,
    $I=\\{I_{20},I_{47},I_{37},I_{86},I_{58},I_{56},I_{54},I_{38},I_{12},I_{39},I_{68},I_{75},I_{66},I_{2},I_{99}\\}$
- Computed the following solution of value 254 for the subproblem:
  $\\{I_{12}, I_{20}, I_{37}, I_{47}, I_{75}\\}$
- Combining
  $\\{I_{5}, I_{7}, I_{8}, I_{15}, I_{50}, I_{84}\\}\cup \\{I_{12}, I_{20}, I_{37}, I_{47}, I_{75}\\}$
- New solution of value 560:
  $\\{I_{5}, I_{7}, I_{8}, I_{12}, I_{15}, I_{20}, I_{37}, I_{47}, I_{50}, I_{75}, I_{84}\\}$

#### Example 2: Different Neighborhoods for the Traveling Salesman Problem

Simply removing a portion of the solution and then trying to fix it is not the
most effective approach. In this section, we will explore various neighborhoods
for the Traveling Salesman Problem (TSP). The geometry of TSP not only permits
advantageous neighborhoods but also offers visually appealing representations.
When you have several neighborhood strategies, they can be dynamically
integrated using an Adaptive Large Neighborhood Search (ALNS).

The image illustrates an optimization process for a tour that needs to traverse
the green areas, factoring in turn costs, within an embedded graph (mesh). The
optimization involves choosing specific regions (highlighted in red) and
calculating the optimal tour within them. As iterations progress, the initial
tour generally improves, although some iterations may not yield any enhancement.
Regions in red are selected due to the high cost of the tour within them. Once
optimized, the center of that region is added to a tabu list, preventing it from
being chosen again.

|                                                                   ![Large Neighborhood Search Geometry Example](https://github.com/d-krupke/cpsat-primer/blob/main/images/lns_pcpp.png)                                                                   |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Large Neighbordhood Search for Coverage Path Planning by repeatedly selecting a geometric region (red) and optimizing the tour within it. The red parts of the tour highlight the changes in the iteration. Read from left to right, and from up to down. |

How can you determine the appropriate size of a region to select? You have two
main options: conduct preliminary experiments or adjust the size adaptively
during the search. Simply allocate a time limit for each iteration. If the
solver does not optimize within that timeframe, decrease the region size.
Conversely, if it does, increase the size. Utilizing exponential factors will
help the size swiftly converge to its optimal dimension. However, it's essential
to note that this method assumes subproblems are of comparable difficulty and
may necessitate additional conditions.

For the Euclidean TSP, as opposed to a mesh, optimizing regions is not
straightforward. Multiple effective strategies exist, such as employing a
segment from the previous tour rather than a geometric region. By implementing
various neighborhoods and evaluating their success rates, you can allocate a
higher selection probability to the top-performing ones. This approach is
demonstrated in an animation crafted by two of my students, Gabriel Gehrke and
Laurenz Illner. They incorporated four distinct neighborhoods and utilized ALNS
to dynamically select the most effective one.

|                                                                                                                                                                         ![ALNS TSP](https://github.com/d-krupke/cpsat-primer/blob/main/images/alns_tsp_compr.gif)                                                                                                                                                                         |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Animation of an Adaptive Large Neighborhood Search for the classical Traveling Salesman Problem. It uses four different neighborhood strategies which are selected randomly with a probability based on their success rate in previous iterations. If you check the logs of the latest (v9.8) version of CP-SAT, it also rates the performance of its LNS-strategies and uses the best performing strategies more often (UCB1-algorithm). |

#### Multi-Armed Bandit: Exploration vs. Exploitation

Having multiple strategies for each iteration of your LNS available is great,
but how do you decide which one to use? You could just pick one randomly, but
this is not very efficient as it is unlikely to select the best one. You could
also use the strategy that worked best in the past, but maybe there is a better
one you have not tried yet. This is the so-called exploration vs. exploitation
dilemma. You want to exploit the strategies that worked well in the past, but
you also want to explore new strategies to find even better ones. Luckily, this
problem has been studied extensively as the
[Multi-Armed Bandit Problem](https://en.wikipedia.org/wiki/Multi-armed_bandit)
for decades, and there are many good solutions. One of the most popular ones is
the Upper Confidence Bound (UCB1) algorithm, which is also used by CP-SAT. In
the following, you can see the a LNS-statistic of the CP-SATs strategies.

```
LNS stats                Improv/Calls  Closed  Difficulty  TimeLimit
       'graph_arc_lns':          5/65     49%        0.26       0.10
       'graph_cst_lns':          4/65     54%        0.47       0.10
       'graph_dec_lns':          3/65     49%        0.26       0.10
       'graph_var_lns':          4/66     55%        0.56       0.10
           'rins/rens':         23/66     39%        0.03       0.10
         'rnd_cst_lns':         12/66     50%        0.19       0.10
         'rnd_var_lns':          6/66     52%        0.36       0.10
    'routing_path_lns':         41/65     48%        0.10       0.10
  'routing_random_lns':         24/65     52%        0.26       0.10
```

We will not dig into the details of the algorithm here, but if you are
interested, you can find many good resources online. I just wanted to make you
aware of the exploration vs. exploitation dilemma and that many smart people
have already thought about it.

> TODO: Continue...
