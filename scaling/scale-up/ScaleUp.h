//
// Created by Ahmed on 15-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUP_H
#define GRAPH_SCALING_TOOL_SCALEUP_H

#include <math.h>
#include "ScaleUpSamplesInfo.h"
#include "../../graph/Graph.h"
#include "../scale-down/Sampling.h"
#include "IdentifierTracker.h"

class ScaleUp {
private:
    Graph* graph;
    Sampling* sampling;
    ScaleUpSamplesInfo* scaleUpSamplesInfo;
    std::string outputFolder;

    bool shouldSampleRemainder(ScaleUpSamplesInfo *scaleUpSamplesInfo, int currentLoopIteration);

    void printScaleUpSetup();

    std::vector<Graph*> createDistinctSamples();

    void createSample(std::vector<Graph*> &samples, float samplingFraction);

public:
    ScaleUp(Graph* graph, Sampling* sampling, ScaleUpSamplesInfo* scaleUpSamplesInfo, std::string outputFolder);

    void executeScaleUp();
};


#endif //GRAPH_SCALING_TOOL_SCALEUP_H
