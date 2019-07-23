//
// Created by Ahmed on 22-7-19.
//

#ifndef GRAPH_SCALING_TOOL_MODEL_H
#define GRAPH_SCALING_TOOL_MODEL_H

#include <string>

class Model {
protected:
    int originalDiameter;
    int numberOfSamples;
    float scalingFactor;

public:
    Model(int originalDiameter, int numberOfSamples, float scalingFactor);

    virtual int getMaxDiameter() = 0;
    virtual std::string getName() = 0;
};


#endif //GRAPH_SCALING_TOOL_MODEL_H
