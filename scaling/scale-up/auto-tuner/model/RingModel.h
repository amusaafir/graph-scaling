//
// Created by Ahmed on 22-7-19.
//

#ifndef GRAPH_SCALING_TOOL_RINGMODEL_H
#define GRAPH_SCALING_TOOL_RINGMODEL_H


#include <string>
#include "Model.h"

class RingModel : public Model {
public:
    RingModel(int originalDiameter, int numberOfSamples, float scalingFactor);
    int getMaxDiameter();
    std::string getName();
};




#endif //GRAPH_SCALING_TOOL_RINGMODEL_H
