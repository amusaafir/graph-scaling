//
// Created by Ahmed on 12-11-17.
//

#include "Edge.h"

Edge::Edge(int source, int target) {
    this->source = source;
    this->target = target;
}

int Edge::getSource() const {
    return source;
}

void Edge::setSource(int source) {
    Edge::source = source;
}

int Edge::getTarget() const {
    return target;
}

void Edge::setTarget(int target) {
    Edge::target = target;
}
