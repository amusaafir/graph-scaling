//
// Created by Ahmed on 5-1-18.
//

#ifndef GRAPH_SCALING_TOOL_WRITESCALEDUPGRAPH_H
#define GRAPH_SCALING_TOOL_WRITESCALEDUPGRAPH_H

#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <sstream>
#include "../graph/Edge.h"
#include "../graph/Graph.h"
#include "../scaling/scale-up/ScaleUpSamplesInfo.h"

class WriteScaledUpGraph {
private:
    std::vector<Graph*> samples;
    std::vector<Edge<std::string>> bridges;
    std::string outputFolderPath;

    void writeEdgesFromSamples(std::ofstream &outputFile) const;
    void writeEdgesFromBridges(std::ofstream &outputFile) const;

public:
    WriteScaledUpGraph(std::string outputFolderPath, std::vector<Graph*> samples, std::vector<Edge<std::string>> bridges);
    void writeToFile(ScaleUpSamplesInfo* scaleUpSamplesInfo);

    std::string createFilename(ScaleUpSamplesInfo *scaleUpSamplesInfo) const;
};


#endif //GRAPH_SCALING_TOOL_WRITESCALEDUPGRAPH_H
