//
// Created by Ahmed on 12-11-17.
//

#include "Scaling.h"
#include "scale-up/topology/StarTopology.h"
#include "scale-up/bridge/RandomBridge.h"

Scaling::Scaling(Graph* graph) {
    this->graph = graph;
}

void Scaling::scaleUp(ScaleUpSamplesInfo* scaleUpSamplesInfo) {
    ScaleUp* scaleUp = new ScaleUp(graph, new TIES(graph), scaleUpSamplesInfo);
    scaleUp->executeScaleUp();
}

void Scaling::scaleDown(float samplingFraction) {
    Sampling* sampling = new TIES(graph);
    sampling->sample(samplingFraction);

    delete(sampling);
}