//
// Created by Ahmed on 12-11-17.
//

#include "Scaling.h"
#include "scale-down/RandomNode.h"
#include "../io/WriteSampledGraph.h"

Scaling::Scaling(Graph* graph, UserInput* userInput) {
    this->graph = graph;
    this->userInput = userInput;
}

void Scaling::scaleUp() {
    ScalingUpConfig* scaleUpSamplesInfo = new ScalingUpConfig(userInput->getScalingFactor(),
                                                              userInput->getSamplingFraction(),
                                                              userInput->getTopology(),
                                                              userInput->getSamplingAlgorithm(graph));

    ScaleUp* scaleUp = new ScaleUp(graph, scaleUpSamplesInfo, userInput->getOutputGraphPath());
    scaleUp->run();

    delete(scaleUp);
    delete(scaleUpSamplesInfo);
}

void Scaling::scaleDown() {
    Sampling* sampling = userInput->getSamplingAlgorithm(graph);
    sampling->run(userInput->getSamplingFraction(), userInput->getOutputGraphPath());

    delete(sampling);
}
