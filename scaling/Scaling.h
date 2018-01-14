//
// Created by Ahmed on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALER_H
#define GRAPH_SCALING_TOOL_SCALER_H

#include "../graph/Graph.h"
#include "scale-down/Sampling.h"
#include "scale-down/TIES.h"
#include "scale-up/ScaleUp.h"

class Scaling {
private:
    Graph* graph;

public:
    Scaling(Graph* graph);
    void scaleUp(ScaleUpSamplesInfo* scaleUpSamplesInfo, std::string outputFolder);
    void scaleDown(float samplingFraction);
};


#endif //GRAPH_SCALING_TOOL_SCALER_H
