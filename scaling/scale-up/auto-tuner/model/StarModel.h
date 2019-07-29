//
// Created by Ahmed on 22-7-19.
//

#ifndef GRAPH_SCALING_TOOL_STARMODEL_H
#define GRAPH_SCALING_TOOL_STARMODEL_H

#include <string>
#include "Model.h"

class StarModel : public Model {
public:
    StarModel(int originalDiameter, int numberOfSamples, float scalingFactor);
    int getMaxDiameter();
    std::string getName();
    Topology* createTopology(Bridge*);
};


#endif //GRAPH_SCALING_TOOL_STARMODEL_H
