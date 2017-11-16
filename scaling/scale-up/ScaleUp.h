//
// Created by Ahmed on 15-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUP_H
#define GRAPH_SCALING_TOOL_SCALEUP_H

#include <math.h>
#include "ScaleUpSamplesInfo.h"
#include "../../graph/Graph.h"
#include "../scale-down/Sampling.h"

class ScaleUp {
private:
    Graph* graph;
    Sampling* sampling;
    float samplingFraction;

    bool shouldApplyRemainderSampling(const ScaleUpSamplesInfo *scaleUpSampleRemainder, int currentLoopIteration) const;

    void printScaleUpSetup(float scalingFactor, const ScaleUpSamplesInfo *scaleUpSamplesInfo) const;

    void createDifferentSamples(ScaleUpSamplesInfo* scaleUpSamplesInfo) const;

public:
    ScaleUp(Graph* graph, Sampling* sampling, float samplingFraction);

    void executeScaleUp(float scalingFactor);
};


#endif //GRAPH_SCALING_TOOL_SCALEUP_H
