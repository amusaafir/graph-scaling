//
// Created by Ahmed on 12-11-17.
//

#include "Scaling.h"

Scaling::Scaling(Graph* graph) {
    this->graph = graph;
}

void Scaling::scaleUp(float scalingFactor, float samplingFraction) {
    ScaleUp* scaleUp = new ScaleUp(graph, new TIES(graph), samplingFraction);
    scaleUp->executeScaleUp(scalingFactor);
}

void Scaling::scaleDown(float samplingFraction) {
    Sampling* sampling = new TIES(graph);
    sampling->sample(samplingFraction);

    delete(sampling);
}
