//
// Created by Ahmed on 15-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUP_H
#define GRAPH_SCALING_TOOL_SCALEUP_H

#include <math.h>
#include "../../graph/Graph.h"
#include "../scale-down/Sampling.h"

class ScaleUp {
private:
    Graph* graph;
    Sampling* sampling;
    float samplingFraction;

public:
    ScaleUp(Graph* graph, Sampling* sampling, float samplingFraction);

    void executeScaleUp(float scalingFactor);

    bool shouldSampleRemainingFraction(int &amountOfSamples, float remainingFraction);

    void createDifferentSamples(int amountOfSamples, float remainingFraction, bool shouldSampleRemainder) const;
};


#endif //GRAPH_SCALING_TOOL_SCALEUP_H
