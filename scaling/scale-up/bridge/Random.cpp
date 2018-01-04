//
// Created by Ahmed on 12-12-17.
//

#include "Random.h"

Random::Random(int numberOfInterconnections, bool forceUndirectedEdges)
        : Bridge(numberOfInterconnections, forceUndirectedEdges) {}

void Random::addBridgesBetweenGraphs(Graph *left, Graph *right) {
}