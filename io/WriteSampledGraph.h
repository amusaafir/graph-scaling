//
// Created by Ahmed on 26-1-18.
//

#ifndef GRAPH_SCALING_TOOL_WRITESAMPLEDGRAPH_H
#define GRAPH_SCALING_TOOL_WRITESAMPLEDGRAPH_H

#include <iostream>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <sstream>
#include <iomanip>

#include "../graph/Graph.h"
#include "WriteGraph.h"

class WriteSampledGraph : public WriteGraph {
private:
    Graph* graph;
    float fraction;

    std::string getFileName();

    void writeGraphEdges(std::ofstream &outputFile);

    void writeToFile();

public:
    WriteSampledGraph(Graph* graph, std::string outputFolderPath, float fraction);
};


#endif //GRAPH_SCALING_TOOL_WRITESAMPLEDGRAPH_H
