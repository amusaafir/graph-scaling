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
    const float MAX_DEVIANCE = 0.1;
    Bridge* bridge;
    int originalDiameter;
    int targetDiameter;
    Node<int>* diameterRoot = NULL;
    Node<int>* currentClosestDiameterNode = NULL;
    int numberOfSampes = 0;
    int iterations; // Number of attempts to find a proper scaled up match

public:
    Autotuner(int originalDiameter, int targetDiameter, int numberOfSamples);
    //SuggestedParameters tuneDiameter();
    SuggestedParameters getNewSuggestion();
    Node<int>* findClosestDiameterNode(Node<int>* node);
    bool isInsideDiameterMargin(int currentDiameter);
    void addNodeToDiameterTree(int diameter, SuggestedParameters suggestedParameters, bool isHeuristic);

    float computeDeviance(int diameter, int targetDiameter);
};


#endif //GRAPH_SCALING_TOOL_AUTOTUNER_H
