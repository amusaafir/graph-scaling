//
// Created by Ahmed on 12-12-17.
//

#include "Bridge.h"

Bridge::Bridge(long long numberOfInterconnections, bool addDirectedBridges) {
        this->numberOfInterconnections = numberOfInterconnections;
        this->addDirectedBridges = addDirectedBridges;
}

long long Bridge::getNumberOfInterconnections() {
        return this->numberOfInterconnections;
}

void Bridge::setNumberOfInterconnections(long long numberOfInterconnections) {
        this->numberOfInterconnections = numberOfInterconnections;
}