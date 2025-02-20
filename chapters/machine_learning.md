## Classical Optimization vs. Machine Learning (and the impact of Quantum Computing)

<a name="chapters-machine-learning"></a>

A prevalent discussion in today's landscape revolves around whether Machine
Learning (ML) or Quantum Computing (QC) could replace classical optimization
methods such as CP-SAT. In this chapter, we will explore the fundamental
differences between Machine Learning and Optimization, demonstrating that these
fields are not interchangeable but rather complementary to each other. By
understanding their distinct strengths and applications, we can better leverage
each approach to solve complex problems effectively.

Following our discussion on these differences, a common question arises: will
Quantum Computing render classical optimization techniques obsolete? To address
this, we will delve into the basic challenges facing Quantum Computing in the
context of Optimization and explain why its impact may be less significant than
often anticipated. This section aims to debunk several myths, some of which are
perpetuated by proponents of Quantum Computing.

Machine Learning excels at **predicting** outcomes based on historical data,
identifying patterns, and making informed guesses, such as estimating the best
solution to an optimization problem based on data patterns. It is adept at
learning and generalizing from data, even if the data is imperfect, provided
there is sufficient quantity. On the other hand, Optimization is powerful for
systematically **searching** for the best solution based on a well-defined
mathematical model, capable of optimizing variables and constraints with
precision, often requiring minimal data. For example, in planning delivery
routes, ML can predict driving times and resource needs based on historical
factors, whereas an optimization solver like CP-SAT is better suited to
determining the most efficient routes by evaluating the interdependencies and
constraints systematically. Both fields enhance decision-making processes when
used together, leveraging ML’s predictive capabilities and Optimization’s
rigorous solution-finding methods.

In the article
[**Four Key Differences Between Mathematical Optimization And Machine Learning**](https://www.forbes.com/councils/forbestechcouncil/2021/06/25/four-key-differences-between-mathematical-optimization-and-machine-learning/),
Edward Rothberg, the CEO of Gurobi, highlights four key differences between
Machine Learning and Optimization:

- **Analytical Focus**: ML is primarily a **predictive** tool that identifies
  patterns in historical data to forecast future events, whereas Optimization is
  a **prescriptive** tool that uses a digital twin of your environment to
  recommend the best possible decisions.
- **Typical Applications**: ML is commonly used for tasks like fraud detection,
  speech recognition, and product recommendations—often consumer-facing
  applications. Optimization is leveraged for operational decision-making in
  areas like production planning, scheduling, and shipment routing.
- **Adaptability**: ML can suffer from "model drift" if the environment changes
  significantly, requiring retraining with new data. Optimization models,
  however, can be updated more seamlessly to reflect changes in real time but
  usually need more upfront effort to build.
- **Maturity**: Both fields have roots tracing back decades, but Optimization
  has largely settled into a "plateau of productivity," while ML is currently at
  the “peak of inflated expectations” and may face a phase of disillusionment
  before stabilizing into broader adoption.

> [!TIP]
>
> You do not have to read this section if you are not interested in my
> arguments. The TL;DR is that neither Machine Learning nor Quantum Computing
> will make CP-SAT (and similar methods) obsolete any time soon, if ever.
> However, Machine Learning is a valuable complement to Optimization.

> :video:
>
> [ML ain't your only hammer: adding mathematical optimisation to the data scientist's toolbox](https://www.youtube.com/watch?v=G0tlyC9Sr3w):
> This 20-minute talk by Dr. Jack Simpson introduces data scientists with a
> machine learning focus to mathematical optimization, highlighting how it
> prescribes optimal decisions under complex constraints and complements ML
> forecasts.

### Using GenAI/LLMs for Optimization

<!-- Concrete example of using ChatGPT for optimizing -->

It is indeed possible to ask ChatGPT (or similar large language models) to solve
certain optimization problems, and in simpler cases (like the Knapsack Problem)
it often succeeds by automatically writing Python code that calls an external
solver (e.g., HiGHs). Although this approach may work for small instances, it
generally runs much slower than a dedicated solver and can introduce subtle
errors. In the example below, ChatGPT took around 10 seconds, whereas a
specialized solver would have solved the same instance almost instantly:

| ![ChatGPT Optimization](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/chatgpt-knapsack_1.png) | ![ChatGPT Optimization Analysis](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/chatgpt-knapsack_2.png) |
| :-----------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: |
|                      Asking ChatGPT to solve a Knapsack Problem, which it does successfully...                      |                                   ...but under the hood, it relies on an external solver.                                    |

<!-- LLMs can reason but only in short turns -->

More advanced versions of ChatGPT (e.g., ChatGPT 4) can tackle somewhat larger
problems but still fall short of solvers like CP-SAT that systematically
evaluate massive search spaces with efficient backtracking and pruning. Large
language models process information sequentially in short "turns", have limited
context windows, and lack specialized heuristics - factors that make them prone
to logical missteps and slower performance as the problem size grows. Repeatedly
generating code or text also creates overhead, and verifying solutions can
require multiple iterations of prompting, further compounding the time cost.

<!-- LLMs are not a replacement for solvers -->

Because of these limitations, LLMs are unlikely to replace robust solvers for
substantial or complex optimization tasks anytime soon. However, current
research on **hybrid AI-OR methods** aims to combine the flexibility of LLMs
(such as quick prototyping and model building) with the powerful search
capabilities of dedicated solvers. As this field evolves, LLMs may increasingly
assist in formulating or refining optimization models while specialized engines
focus on the computationally intensive search. We will revisit these
possibilities in the upcoming section on "Model Building with GenAI/LLMs".

<!-- Further reading -->

To dig a little deeper into the limitations of LLMs for optimization, consider
the following articles:

- [LLM-ify me - Optimization edition](https://oberdieck.dk/p/llm-in-optimization/):
  A blog post exploring the potential of LLMs for optimization.
- [Mind Evolution and the frontier of LLM-based optimization solvers](https://open.substack.com/pub/feasible/p/64-mind-evolution-and-the-frontier?r=49480z&utm_campaign=post&utm_medium=email):
  An article from the Feasible newsletter discussing the potential of LLMs for
  optimization.
- [Why Solving Multi-agent Path Finding with Large Language Models has not Succeeded Yet](https://arxiv.org/pdf/2401.03630)
- [Look Further Ahead: Testing the Limits of GPT-4 in Path Planning](https://arxiv.org/pdf/2406.12000)
- [Extracting Problem Structure with LLMs for Optimized SAT Local Search](https://arxiv.org/pdf/2501.14630):
  This paper demonstrates how an LLM can be used to automatically generate
  **start heuristics** from Python code that models a problem as a SAT instance
  using the PySAT library. These start heuristics provide the **initial phase**
  for a SAT solver, helping it begin the search from a more promising assignment
  and enabling it to solve additional instances that would remain unsolved
  otherwise.

### Reinforcement Learning

In some cases, an actually viable alternative to optimization solvers like
CP-SAT is Reinforcement Learning. Reinforcement learning can be highly effective
for optimizing very complex problems that are too intricate to model with simple
mathematical formulations. During my dissertation, I applied reinforcement
learning to optimize a multi-agent system where developing a comprehensive
mathematical model was infeasible due to the excessive number of variables and
constraints required to capture the system’s dynamics. Additionally, manually
crafted heuristics proved unable to compete with the performance of a
straightforward reinforcement learning agent, which I implemented with minimal
effort using
[Stable Baselines](https://stable-baselines.readthedocs.io/en/master/). However,
despite reinforcement learning agents being capable of performing many more
iterations to learn about the solution space compared to large language models
(LLMs), they still lack the structured and efficient search capabilities
inherent in classical solvers. Furthermore, designing an appropriate reward
function for reinforcement learning can be challenging and often involves
significant trial and error.

#### Combining Machine Learning and Optimization

Instead of trying to replace optimization solvers like CP-SAT with machine
learning, a more promising approach is to combine the two fields. This can be
achieved in several ways, such as using machine learning to predict parameters
for optimization problems, building optimization proxies, or integrating machine
learning within solvers to guide internal decisions. These methods leverage the
strengths of both fields to enhance decision-making processes and improve
solution quality.

##### Predict-Then-Optimize Variants

<!-- Basic Variant-->

Predict-Then-Optimize is a framework that combines machine learning with
optimization. The basic version involves two simple steps: predict and then
optimize. First, a standard ML model (e.g., a regression) is trained to estimate
unknown parameters of the optimization problem, such as costs or demands, using
conventional loss functions like mean squared error. Next, these predicted
parameters are passed to an optimization solver (such as CP-SAT), which produces
a best-possible solution—say, a schedule, route, or resource allocation—based on
the estimated inputs. This approach is effective when the model’s predictive
accuracy is reasonably high, but it can falter if small errors in the
predictions cause large downstream impacts on the decision.

<!-- Improved Variants -->

To improve decision quality, more advanced, "decision-focused" variants of
Predict-Then-Optimize incorporate the solver's objective directly into the ML
training process. Rather than merely minimizing standard predictive error, these
methods seek to minimize "regret"—the gap between the cost of the solver's
solution under predicted parameters and the cost of the true optimal solution.
By repeatedly using the solver (or an approximation) during training, they
compute how prediction mistakes translate into suboptimal decisions, and feed
this information back to the ML model's parameters. As a result, the model
learns to predict in a way that preserves or improves the final solution's
quality, even if the raw predictive accuracy on each parameter is not perfectly
precise.

<!-- References -->

Check out
[this great lecture by Elias Khalil](https://www.youtube.com/watch?v=pZqm-i57gxk)
which was part of a summer school.

> [!TIP]
>
> Uncertainty and unknown parameters frequently arise in optimization problems.
> While the **predict-then-optimize** framework offers a straightforward
> approach, its more advanced variants can help address some of its limitations.
> For even greater robustness, techniques such as **robust optimization**,
> **stochastic programming**, and others provide more effective solutions,
> especially in highly uncertain environments, though they come with increased
> complexity. Numerous research studies explore these and additional methods,
> with the optimal choice depending heavily on the specific application.

##### Optimization Proxies

<!-- basic idea of optimization proxies -->

An optimization proxy is a machine learning model trained to approximate the
input-output behavior of an optimization solver, enabling near-instant decisions
once deployed. Typically, one gathers a large set of problem instances, solves
them offline using a traditional solver to generate "ground truth" solutions,
and then trains an ML model to map inputs (e.g., demands, costs) to outputs (the
solver's decisions). In real-time or large-scale simulation settings—such as
power systems, scheduling, or routing—using the learned proxy bypasses the usual
heavy solve times, providing solutions (or near-solutions) almost instantly.
Because the training data comes directly from the solver, no manual labeling is
required; any number of instances can be generated offline.

<!-- Pros and Cons -->

However, optimization proxies excel only when repeatedly solving relatively
similar models under stable conditions, and they can struggle when complex
feasibility constraints or drastically changing problem parameters arise. Though
they have found very useful applications in areas like power networks, many
other optimization contexts involve too many shifting variables or intricate
constraints for a simple proxy to handle reliably. Substantial changes to the
problem's structure often require retraining or extensive model adaptation. As a
result, while proxies can be a powerful add-on to speed up certain classes of
problem instances, they are by no means a universal replacement for classical
solvers.

<!-- Further Reading -->

A [short 4min explanation](https://www.youtube.com/watch?v=UAwEZi56cb8) is given
by Pascal Van Hentenryck. And a longer version can be found
[here](https://www.youtube.com/watch?v=NlwxEGtw4QY).

##### Model Building with GenAI/LLMs

An increasingly valuable application of LLMs (or generative AI) in optimization
is assisting with the model building process. Rather than trying to solve an
optimization problem directly, which is often ineffective, these models can help
set up the initial model that a specialized solver will handle. Many
optimization models share similar structures and differ only in certain details,
making LLMs useful for producing core components. Tools like GitHub Co-Pilot can
already generate complex parts of a model, yet they may also introduce subtle
errors that are hard to detect, such as off by one mistakes, inverted
constraints, or swapped indices. It is therefore best to use LLMs as a source of
inspiration and verify their output carefully; otherwise, the time you save
coding might be spent on debugging.

Moreover, your optimization model often represents the backbone of a real world
problem, so it is crucial to understand how it aligns with the underlying
operational or business context. While LLMs can expedite coding, they cannot
replace the human expertise needed to approximate and simplify real world
complexities. If the scenario is critical, you may not want to rely too heavily
on an AI-based approach at this stage. That said, several research projects and
commercial products are already exploring this idea:

1. [A Research Project at Stanford](https://web.stanford.edu/~udell/project-modeling.html)
2. [Gurobi AI Modeling](https://gurobi-ai-modeling.readthedocs.io/en/latest/index.html)
   ([Quick Overview](https://www.youtube.com/watch?v=8hr_23zdRV4))
3. [Quantagonia](https://www.quantagonia.com/decisionai) provides a solver you
   can interact with via chat
4. [Robust and Adaptive Optimization under a Large Language Model Lens](https://arxiv.org/pdf/2501.00568):
   A research paper exploring using LLMs for robust optimization.

##### Learning instead of Guessing

Machine learning can also be integrated _inside_ solvers, guiding internal
decisions such as branching strategies, cut selection, or matrix scaling. Rather
than relying on hand-tuned rules alone, the solver gathers data across diverse
problem instances and learns which algorithmic choices best reduce run time or
improve numerical stability. For instance, a model can predict whether "local"
or "global" cutting planes will be more effective on a given instance, or
whether certain scaling methods will avoid ill-conditioned bases. By treating
these decisions as regression tasks - estimating speedup or stability
improvements - machine learning lets the solver adapt and self-tune, ultimately
performing better on a wide spectrum of problems without sacrificing generality.

This
[lecture by Timo Berthold (FICO)](https://www.youtube.com/watch?v=xYKNH3Pqq9A)
gives a good overview of the topic. It has been part of the CO@Work 2024 summer
school, which I actually attended and consider to be absolutely amazing and if
you get the chance to attend, I highly recommend it (as long as you already are
on PhD-level, as it is intense).

A three-hour tutorial by various experts on the topic (including Elias Khalil
and Andrea Lodi), can be found
[here](https://www.youtube.com/watch?v=XVLd7hf6y6M&list=LL&index=20). I highly
recommend watching it, if you are interested in the topic as I had many "aha"
moments during this tutorial.

> :reference:
>
> - [Machine learning augmented branch and bound for mixed integer linear programming](https://link.springer.com/article/10.1007/s10107-024-02130-y):
>   This recent paper from 2024 surveys ideas for using machine learning to
>   improve branch-and-bound solvers for mixed-integer linear programming.

### Why Quantum Computing Will (Probably) Not Have a Big Impact on Optimization

Before closing this chapter, let us also discuss the impact of Quantum Computing
on Optimization, as this is a question I am frequently asked, usually directly
after having explained why ChatGPT cannot replace CP-SAT.

<!-- Dangerous Claims -->

You often hear claims that quantum computing will revolutionize optimization,
especially regarding the Traveling Salesman Problem (TSP). These claims
frequently state that a new quantum algorithm can solve TSP (a challenging yet
practically important combinatorial optimization problem) for small instance
sizes, while a classical computer would supposedly need billions of years to
handle as few as 20 nodes. (In reality, some published papers still only address
four nodes.) Unfortunately, such claims usually hinge on a theoretical
worst-case runtime of around $O(n^2 2^n)$ for the TSP, which is not
representative of how the problem is handled in practice.

> :reference:
>
> For an accessible and insightful discussion on the myths and exaggerated
> expectations surrounding quantum computing, consult the freely available book
> [_What You Shouldn't Know About Quantum Computers_](https://arxiv.org/abs/2405.15838)
> by Chris Ferrie. This resource offers an excellent opportunity to critically
> examine common misconceptions often perpetuated by science fiction and popular
> science literature.

<!-- Worst-Case vs. Real-World -->

Although the best-known quantum algorithm runs in $O(1.728^n)$, which is
somewhat better than $O(n^2 2^n)$, it remains exponential and grows very
quickly. Since the TSP is NP-hard, it is unlikely we will ever discover a
(quantum or classical) algorithm whose worst-case runtime does not explode with
instance size. Moreover, real-world or average-case performance can be vastly
different from worst-case performance. In fact, TSP can already be solved
effectively for large instances on classical computers, so many bold claims
about quantum computing's purported advantages in optimization can be misleading
or simply incorrect.

<!-- TL;DR -->

To date, there is no strong evidence suggesting that quantum computing will have
a major impact on optimization. Many experts believe that, at best, quantum
computing might offer only a modest performance advantage in this domain, though
it may have significant implications for cryptology.

> [!TIP]
>
> At a recent consortium meeting, a consultant from a logistics optimization
> firm made a particularly noteworthy remark. According to them, even with the
> advent of a fully functional quantum computer equipped with thousands of
> flawless qubits, its primary utility would lie in marketing. Beyond that, it
> would be little more than "metal trash," offering no tangible value for their
> customers' use cases. This statement prompted humorous objections from quantum
> experts regarding the "metal trash" comment. Nevertheless, the broader
> sentiment that quantum computing is overhyped in the optimization domain was
> widely shared.

<!-- Quantum Computers are currently worse than Pen&Paper -->

To put the Traveling Salesman Problem's difficulty into historical perspective:
In the mid-20th century, the TSP attracted attention through challenges offering
substantial prize money for solving relatively small instances of 33 to 49
cities - already far larger than those currently tackled by many quantum
researchers. As Newsweek reported in **1954**:

> "By an ingenious application of linear programming - a mathematical tool
> recently used to solve production-scheduling problems - it took only a few
> weeks for the California experts to calculate by hand the shortest route to
> cover the 49 cities."

Remarkably, they not only found the shortest route, but also proved it was the
shortest, all by hand in the 1950s. You can read more details in
[this blog post](https://www.math.uwaterloo.ca/tsp/us/history.html). Nowadays,
there is even
[an iPhone app](https://apps.apple.com/us/app/concorde-tsp/id498366515) that can
solve instances of around 1,000 TSP cities to optimality in mere seconds.
[Larger-scale instances](https://www.math.uwaterloo.ca/tsp/optimal/index.html)
(up to 85,900 nodes, solved in 2006) and near-optimal solutions for millions of
cities illustrate how far classical methods have come. In that sense, quantum
computers have a considerable way to go before they can beat even a human with
pen and paper, let alone a classical computer.

<!-- Anecdote of a wrong complexity estimate leading to financial loss -->

Misjudging the difficulty of certain combinatorial problems famously led one
puzzle inventor into unexpected financial trouble. His creation,
[_The Eternity Puzzle_](https://en.wikipedia.org/wiki/Eternity_puzzle), offered
a £1,000,000 prize to anyone who could solve it—a sum that some early estimates
suggested might never be claimed in a human lifespan, given the vast number of
possible configurations. However, instead of brute-forcing every possibility,
two Cambridge mathematicians employed advanced techniques to dramatically narrow
the search space. Similarly, CP-SAT leverages a variety of behind-the-scenes
strategies to tackle complex problems more efficiently than most would
imagine—indeed, many of the methods used to solve The Eternity Puzzle resemble
what CP-SAT does in a more general way.

Thanks to these refined methods, the puzzle was solved in under 18 months - much
sooner than anticipated. Although rumors circulated that the puzzle's inventor,
Christopher Monckton, was forced to sell his mansion to pay the prize, this
appears to have been more of a publicity stunt; in reality, he could afford to
cover the loss. He later released a successor puzzle, which has proven far more
difficult and remains unsolved.

<!-- Basic Quantum Computing and Advantage -->

Quantum computers exploit superposition and entanglement to evaluate multiple
solutions in parallel. However, they do not simply evaluate all $n!$
permutations at once and automatically return the best one - an assumption that,
unfortunately, is still common. Once you measure the output of a quantum
algorithm, you effectively collapse the wavefunction, ending up with a single
measured state. Thus, careful algorithm design is required to boost the
probability of measuring the correct solution, often requiring many repeated
measurements. A relevant quantum algorithm in this context is Grover's
Algorithm, which offers a quadratic speedup in unstructured searches (e.g., from
$O(2^n)$ to $O(\sqrt{2^n})$). However, TSPs and other optimization problems have
enough internal structure in practice that classical approaches can exploit it,
often rendering Grover's quadratic speedup unimpressive by comparison.

<!-- Shortcomings of Quantum Computers compared to Classical Computers -->

Quantum computers also come with significant drawbacks relative to classical
computers. Their quantum state collapses upon measurement, preventing techniques
like "early abort" or classical branch-and-bound pruning, where large portions
of the search space can be discarded based on intermediate results. Classical
algorithms rely on this kind of dynamic pruning to speed up the search. Quantum
computing sacrifices flexibility for parallel evaluation—an advantage in
unstructured or purely guesswork-based problems, but most practical optimization
problems have exploitable structure. While there may be ways around these
limitations and, in theory, quantum computers are more powerful than classical
ones, whether this yields substantial real-world benefits for optimization
remains unclear.

<!-- Comparison to GP-GPU -->

In some sense, quantum computers face challenges reminiscent of GPU computing:
both process many data points in parallel (Single-Instruction, Multiple-Data),
but lose out on some flexibility since every thread must follow identical
instructions. Even for GPUs, applying them effectively to large-scale
optimization has proven tricky, though progress continues.

<!-- Some Resources -->

If you want more perspectives on the topic, the following articles may be of
interest:

- [Challenges and Opportunities in Quantum Optimization](https://arxiv.org/pdf/2312.02279):
  This preprint gives on 72 pages an extensive and well-written overview of the
  state of the art in quantum optimization. While the conclusion sounds quite
  optimistic, the paper is very clear about the current limitations and
  challenges. while it caters a scientific audience, it is still quite
  accessible.
- [Understanding instance hardness for optimisation algorithms: Methodologies, open challenges and post-quantum implications](https://www.sciencedirect.com/science/article/pii/S0307904X2500040X):
  This paper states that "it seems likely that there will be no quantum
  advantage for the TSP" due to challenges in tuning the parameters of QUBOs to
  even yield feasible solutions. However, it considers unconstrained
  optimization problems such as the MAX-CUT to have a higher chance of
  benefiting from quantum computing.
- [Disentangling Hype from Practicality: On Realistically Achieving Quantum Advantage](https://cacm.acm.org/research/disentangling-hype-from-practicality-on-realistically-achieving-quantum-advantage/#R3)
  Gives strong arguments why it can be hard to achieve quantum advantage in
  practice, especially why quadratic speedup algorithms like Grover's Algorithm
  will not suffice.
- [Quantum advantage for NP approximation? For REAL this time?](https://scottaaronson.blog/?p=8375):
  A blog post by Scott Aaronson, a well-known quantum computing expert, offering
  a critical take on the QAOA algorithm and its claimed advantages.
- [Challenges and opportunities in quantum optimization](https://www.nature.com/articles/s42254-024-00770-9):
  A balanced discussion by a group of researchers, highlighting potential
  opportunities without making unfounded claims.
- [Quantum Annealing versus Digital Computing: An Experimental Comparison](https://www.researchgqate.net/publication/353155344_Quantum_Annealing_versus_Digital_Computing_An_Experimental_Comparison):
  This paper compares quantum annealing to classical computing for optimization
  problems and found no indication of a quantum advantage.

<!-- Disclaimer that I am not an expert -->

> [!WARNING]
>
> I do not claim to be an expert in quantum computing, so please interpret my
> remarks with appropriate caution. This book is an open-source project;
> therefore, if you have any corrections or suggestions to help improve the
> material for the community, feel free to open an issue or submit a pull
> request on [GitHub](https://github.com/d-krupke/cpsat-primer/).
