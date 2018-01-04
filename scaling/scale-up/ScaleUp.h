//
// Created by Ahmed on 15-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUP_H
#define GRAPH_SCALING_TOOL_SCALEUP_H

#include <math.h>
#include "helper/ScaleUpSamplesInfo.h"
#include "../../graph/Graph.h"
#include "../scale-down/Sampling.h"
#include "helper/IdentifierTracker.h"

class ScaleUp {
private:
    Graph* graph;
    Sampling* sampling;
    float samplingFraction;

    bool shouldSampleRemainder(ScaleUpSamplesInfo *scaleUpSampleRemainder, int currentLoopIteration);

    void printScaleUpSetup(float scalingFactor, const ScaleUpSamplesInfo *scaleUpSamplesInfo);

    std::vector<Graph*> createDistinctSamples(ScaleUpSamplesInfo *scaleUpSamplesInfo);

    void createSample(IdentifierTracker *identifierTracker, std::vector<Graph*> &samples, float samplingFraction);

public:
    ScaleUp(Graph* graph, Sampling* sampling, float samplingFraction);

    void executeScaleUp(float scalingFactor);
};


#endif //GRAPH_SCALING_TOOL_SCALEUP_H
