//
// Created by Ahmed on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALER_H
#define GRAPH_SCALING_TOOL_SCALER_H

#include "../graph/Graph.h"
#include "scale-down/Sampling.h"
#include "scale-down/TIES.h"
#include "scale-up/ScaleUp.h"

class ScalingManager {
private:
    Graph* graph;

public:
    ScalingManager(Graph* graph);
    void scaleUp(float scalingFactor, float samplingFraction);
    void scaleDown(float samplingFraction);
};


#endif //GRAPH_SCALING_TOOL_SCALER_H
