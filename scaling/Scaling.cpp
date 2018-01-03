//
// Created by Ahmed on 12-11-17.
//

#include "Scaling.h"

ScalingManager::ScalingManager(Graph* graph) {
    this->graph = graph;
}

void ScalingManager::scaleUp(float scalingFactor, float samplingFraction) {
    ScaleUp* scaleUp = new ScaleUp(graph, new TIES(graph), samplingFraction);
    scaleUp->executeScaleUp(scalingFactor);
}

void ScalingManager::scaleDown(float samplingFraction) {
    Sampling* sampling = new TIES(graph);
    sampling->sample(samplingFraction);

    delete(sampling);
}
