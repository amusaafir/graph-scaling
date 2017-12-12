//
// Created by Ahmed on 12-12-17.
//

#include "Random.h"

Random::Random(Graph *left, Graph *right, int numberOfInterconnections, bool forceUndirectedEdges)
        : Bridge(left, right, numberOfInterconnections, forceUndirectedEdges) {}

void Random::addBridges() {
}