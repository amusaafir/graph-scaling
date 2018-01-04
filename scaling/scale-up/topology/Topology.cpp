//
// Created by Ahmed on 12-12-17.
//

#include "Topology.h"

Topology::Topology(std::vector<Graph*> samples, Bridge* bridge) {
    this->samples = samples;
    this->bridge = bridge;
}