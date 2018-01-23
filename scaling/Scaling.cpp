//
// Created by Ahmed on 12-11-17.
//

#include "Scaling.h"
#include "scale-up/topology/StarTopology.h"
#include "scale-up/bridge/RandomBridge.h"

Scaling::Scaling(Graph* graph) {
    this->graph = graph;
}

void Scaling::scaleUp(ScalingUpConfig* scaleUpSamplesInfo, std::string outputFolder) {
    ScaleUp* scaleUp = new ScaleUp(graph, new TIES(graph), scaleUpSamplesInfo, outputFolder);
    scaleUp->executeScaleUp();

    delete(scaleUp);
}

void Scaling::scaleDown(float samplingFraction, std::string outputFolder) {
    Sampling* sampling = new TIES(graph);
    sampling->sample(samplingFraction);

    delete(sampling);
}
