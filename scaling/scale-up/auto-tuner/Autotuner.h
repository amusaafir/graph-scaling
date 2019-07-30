//
// Created by Ahmed on 30-3-19.
//

#ifndef GRAPH_SCALING_TOOL_AUTOTUNER_H
#define GRAPH_SCALING_TOOL_AUTOTUNER_H


#include <map>
#include "../../../graph/Graph.h"
#include "Node.h"
#include "SuggestedParameters.h"
#include "model/Model.h"

/**
 * TODO: Each graph property should be a child of this class, e.g., DiameterAutotuner
 */
class Autotuner {
private:
    Model* topologies[4] = {};
    Bridge* bridge;
    int originalDiameter;
    Node<int>* diameterRoot = NULL;
    int numberOfSampes = 0;
    int iterations; // Number of attempts to find a proper scaled up match

public:
    Autotuner(int originalDiameter, int numberOfSamples);
    //SuggestedParameters tuneDiameter();
    SuggestedParameters findClosestMatch();
    bool isInsideDiameterMargin(int currentDiameter);
    void addNodeToDiameterTree(int diameter, SuggestedParameters suggestedParameters, bool isHeuristic);
};


#endif //GRAPH_SCALING_TOOL_AUTOTUNER_H
