//
// Created by Ahmed on 12-11-17.
//

#include "Scaling.h"
#include "scale-down/RandomNode.h"

Scaling::Scaling(Graph* graph, UserInput* userInput) {
    this->graph = graph;
    this->userInput = userInput;
}

void Scaling::scaleUp() {
    ScalingUpConfig* scaleUpSamplesInfo = new ScalingUpConfig(userInput->getScalingFactor(),
                                                              userInput->getSamplingFraction(),
                                                              userInput->getTopology());

    ScaleUp* scaleUp = new ScaleUp(graph, new TIES(graph), scaleUpSamplesInfo, userInput->getOutputGraphPath());
    scaleUp->run();

    delete(scaleUp);
    delete(scaleUpSamplesInfo);
}

void Scaling::scaleDown() {
    Sampling* sampling = new TIES(graph);
    sampling->run(userInput->getSamplingFraction(), userInput->getOutputGraphPath());

    delete(sampling);
}
