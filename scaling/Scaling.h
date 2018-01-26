//
// Created by Ahmed on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALER_H
#define GRAPH_SCALING_TOOL_SCALER_H

#include "../graph/Graph.h"
#include "scale-down/Sampling.h"
#include "scale-down/TIES.h"
#include "scale-up/ScaleUp.h"
#include "../io/user-input/UserInput.h"

class Scaling {
private:
    Graph* graph;
    UserInput* userInput;

public:
    Scaling(Graph* graph, UserInput* userInput);
    void scaleUp();
    void scaleDown();
};


#endif //GRAPH_SCALING_TOOL_SCALER_H
