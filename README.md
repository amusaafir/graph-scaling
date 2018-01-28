# Sample Spray (graph scaling tool)
#### A Sampling-based Method for Scaling Graph Datasets

This tool allows users to scale a graph up or down using sampling as a basis for both operations.

## Scaling Guidelines and Results



## Usage


### Compilation

TODO


###  Single thread and machine

Running `graph_scaling_tool` without any parameter should prompt the user several input requests accordingly, including whether a scaling up or scaling down operation should be performed.

To perform one of the two operations directly, the user must specify at least:

- The input graph path: **`-i`** (e.g., `-i /home/user/graph.csv`). Note that this can be any format, as long as the input graph is an edge list (where each vertex is represented numerically)
- The output path: **`-o`** (e.g., `-o /home/user/`)


To directly perform a **scaling down** operation, the **`-s`** parameter is required to specify the preferred sample size (based on the number of vertices). This parameter should be between 0 and 1. 

**Example scaling down:** `./graph_scaling_tool -i /home/user/graph.txt -o /home/user/graph_output -s 0.4`

To directly perform a **scaling up** operation, the following parameters are required:

- Scaling factor **`-u`** (based on the number of vertices)
- Sample size **`-s`** for each sampled graph. This value should be between 0 and 1.
- Topology **`-t`** either: `star`, `chain`, `ring` or `fullyconnected`
- Bridging type **`-b`**, either: `random` or `high` (degrees)
- Number of interconnections between each sample: **`-n`**
- Force directed bridges **`-d`**. Setting this to true would add directed bridges.

**Example scaling up:**
`./graph_scaling_tool -i /home/user/graph.txt -o /home/user/graph_output -u 3.0 -s 0.5 -d false -n 10 -b random -t star
`


### Parallel & distributed

It is also possible to run this tool in a parallel and distributed manner.

TODO: Explanation


