# Sample Spray (graph scaling tool)
#### A Sampling-based Method for Scaling Graph Datasets

This tool allows users to scale a graph up or down using sampling as a basis for both operations.

## Scaling Guidelines and Results

- Refer to the [scaling up guidelines](scaling-guidelines.pdf) to target certain property changes whenever scaling up a graph.

- The [scaling-results](scaling-results.pdf) document shows the results of several scaling operations conducted on different graph datasets.

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


### Parallel & distributedMake with:

Switch to the 'distributed' branch; compile the project using:

    make release

Run on a distributed cluster with:

    prun -np <number of compute nodes> -v -1 -reserve <reservation id> -sge-script $PRUN_ETC/prun-openmpi sample <input file> <output file> <sampling fraction> <CPU mem limit (bytes, per node)> <GPU mem limit (bytes, per node)>

Example with 2 nodes, "wiki" input file, 0.5 sampling fraction, 32 GB CPU memory and 12 GB GPU memory limits:
    prun -np 2 -v -1 -reserve <reservation id> -sge-script $PRUN_ETC/prun-openmpi sample OUTPUT_PATH 0.5 34359738368 12884901888

Note that the input file must be an edge list, ordered by source vertex, and then by destination vertex. The number of vertices and edges in the graph must be denoted at the beginning of the file, on a separate rule, like:
"! nVertices nEdges" (without quotes)
The input graph should not have 'missing' vertices (vertices without incoming/outgoing edges and are therefore not present in the input file). 



