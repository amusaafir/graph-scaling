//
// Created by Ahmed on 22-1-18.
//

#include "UserInput.h"

UserInput::UserInput() {
}

UserInput::~UserInput() {
    delete(topologies['s']);
    delete(topologies['c']);
    delete(topologies['r']);
    delete(topologies['f']);

    delete(bridge);
}

void UserInput::initTopologies() {
    topologies['s'] = new StarTopology(bridge);
    topologies['c'] = new ChainTopology(bridge);
    topologies['r'] = new RingTopology(bridge);
    topologies['f'] = new FullyConnectedTopology(bridge);
}