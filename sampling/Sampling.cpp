//
// Created by Ahmed on 14-11-17.
//

#include "Sampling.h"

Sampling::Sampling(Graph* graph) {
    this->graph = graph;
}

int Sampling::getRequiredVerticesFraction(float fraction) {
    return this->graph->getVertices().size() * fraction;
}

int Sampling::getRequiredEdgesFraction(float fraction) {
    return this->graph->getEdges().size() * fraction;
}

int Sampling::getRandomIntBetweenRange(int min, int max) {
    std::random_device seed;
    std::mt19937 engine(seed());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(engine);
}