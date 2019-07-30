//
// Created by Ahmed on 23-7-19.
//

#ifndef GRAPH_SCALING_TOOL_SUGGESTEDPARAMETERS_H
#define GRAPH_SCALING_TOOL_SUGGESTEDPARAMETERS_H


#include "../topology/Topology.h"

class SuggestedParameters {
public:
    Topology* topology;

    std::string getParameterStringRepresentation() {
        return topology->getName()
               + ":" + topology->getBridge()->getName()
               + ":" + std::to_string(topology->getBridge()->getNumberOfInterconnections());
    }
};


#endif //GRAPH_SCALING_TOOL_SUGGESTEDPARAMETERS_H
