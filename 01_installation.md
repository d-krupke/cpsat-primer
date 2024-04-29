<!-- EDIT THIS PART VIA 01_installation.md -->

<a name="01-installation"></a>

## Installation

We are using Python 3 in this primer and assume that you have a working Python 3
installation as well as the basic knowledge to use it. There are also interfaces
for other languages, but Python 3 is, in my opinion, the most convenient one, as
the mathematical expressions in Python are very close to the mathematical
notation (allowing you to spot mathematical errors much faster). Only for huge
models, you may need to use a compiled language such as C++ due to performance
issues. For smaller models, you will not notice any performance difference.

The installation of CP-SAT, which is part of the OR-Tools package, is very easy
and can be done via Python's package manager
[pip](https://pip.pypa.io/en/stable/).

```shell
pip3 install -U ortools
```

This command will also update an existing installation of OR-Tools. As this tool
is in active development, it is recommended to update it frequently. We actually
encountered wrong behavior, i.e., bugs, in earlier versions that then have been
fixed by updates (this was on some more advanced features, do not worry about
correctness with basic usage).

I personally like to use [Jupyter Notebooks](https://jupyter.org/) for
experimenting with CP-SAT.

### What hardware do you need?

It is important to note that for CP-SAT usage, you do not need the capabilities
of a supercomputer. A standard laptop is often sufficient for solving many
problems. The primary requirements are CPU power and memory bandwidth, with a
GPU being unnecessary.

In terms of CPU power, the key is balancing the number of cores with the
performance of each individual core. CP-SAT leverages all available cores,
implementing different strategies on each.
[Depending on the number of cores, CP-SAT will behave differently](https://github.com/google/or-tools/blob/main/ortools/sat/docs/troubleshooting.md#improving-performance-with-multiple-workers).
However, the effectiveness of these strategies can vary, and it is usually not
apparent which one will be most effective. A higher single-core performance
means that your primary strategy will operate more swiftly. I recommend a
minimum of 4 cores and 16GB of RAM.

While CP-SAT is quite efficient in terms of memory usage, the amount of
available memory can still be a limiting factor in the size of problems you can
tackle. When it came to setting up our lab for extensive benchmarking at TU
Braunschweig, we faced a choice between desktop machines and more expensive
workstations or servers. We chose desktop machines equipped with AMD Ryzen 9
7900 CPUs (Intel would be equally suitable) and 96GB of DDR5 RAM, managed using
Slurm. This decision was driven by the fact that the performance gains from
higher-priced workstations or servers were relatively marginal compared to their
significantly higher costs. When on the road, I am often still able to do stuff
with my old Intel Macbook Pro from 2018 with an i7 and only 16GB of RAM, but
large models will overwhelm it. My workstation at home with AMD Ryzen 7 5700X
and 32GB of RAM on the other hand rarely has any problems with the models I am
working on.

For further guidance, consider the
[hardware recommendations for the Gurobi solver](https://support.gurobi.com/hc/en-us/articles/8172407217041-What-hardware-should-I-select-when-running-Gurobi-),
which are likely to be similar. Since we frequently use Gurobi in addition to
CP-SAT, our hardware choices were also influenced by their recommendations.
