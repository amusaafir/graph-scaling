//
// Created by Ahmed on 26-1-18.
//

#ifndef GRAPH_SCALING_TOOL_WRITESAMPLEDGRAPH_H
#define GRAPH_SCALING_TOOL_WRITESAMPLEDGRAPH_H

#include <iostream>
#include <string>
#include "../graph/Graph.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <sstream>

// TODO: Re-use WriteScaledUpGraph; extract original input graph name

class WriteSampledGraph {
private:
    Graph* graph;
    std::string outputFolderPath;
    float fraction;

    std::string getFileName();

    void writeGraphEdges(std::ofstream &outputFile);
public:
    WriteSampledGraph(Graph* graph, std::string outputFolderPath, float fraction);

    void writeToFile();
};


#endif //GRAPH_SCALING_TOOL_WRITESAMPLEDGRAPH_H
