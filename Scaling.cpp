//
// Created by Ahmed on 12-11-17.
//

#include "Scaling.h"
#include "sampling/Sampling.h"
#include "sampling/TIES.h"

Scaling::Scaling() {
    initScaling();
}

Scaling::~Scaling() {
    delete(graphLoader);
}

void Scaling::initScaling() {
    Graph* graph = graphLoader->loadGraph("/home/aj/Documents/graph_datasets/facebook_combined.txt");

    Sampling* sampling = new TIES(graph);
    sampling->sample(0.3);

    delete(graph);
}
