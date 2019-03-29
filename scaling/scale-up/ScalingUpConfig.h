//
// Created by Ahmed on 16-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H
#define GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H

#include <math.h>
#include "topology/Topology.h"
#include "bridge/Bridge.h"

class ScalingUpConfig {
private:
    Topology* topology;
    int amountOfSamples;
    float remainder;
    float scalingFactor;
    float samplingFraction;

    void determineAdditionalSampling();

public:
    ScalingUpConfig(float scalingFactor, float samplingFraction, Topology* topology);

    bool hasSamplingRemainder() const;

    float getRemainder() const;

    void setRemainder(float remainder);

    int getAmountOfSamples() const;

    void setAmountOfSamples(int amountOfSamples);

    float getScalingFactor();

    float getSamplingFraction();

    Topology* getTopology();

    void setTopology(Topology* topology);
};


#endif //GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H
