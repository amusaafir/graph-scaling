//
// Created by Ahmed on 16-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H
#define GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H

#include <math.h>
#include "topology/Topology.h"
#include "bridge/Bridge.h"

class ScaleUpSamplesInfo {
private:
    Topology* topology;
    int amountOfSamples;
    float remainder;
    float scalingFactor;
    float samplingFraction;

    void determineAdditionalSampling();

public:
    ScaleUpSamplesInfo(Topology* topology, float scalingFactor, float samplingFraction);

    bool hasSamplingRemainder() const;

    float getRemainder() const;

    void setRemainder(float remainder);

    int getAmountOfSamples() const;

    void setAmountOfSamples(int amountOfSamples);

    float getScalingFactor();

    float getSamplingFraction();
};


#endif //GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H
