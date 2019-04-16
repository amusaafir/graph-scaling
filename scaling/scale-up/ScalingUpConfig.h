//
// Created by Ahmed on 16-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H
#define GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H

#include <math.h>
#include <iostream>
#include "topology/Topology.h"
#include "bridge/Bridge.h"
#include "../scale-down/Sampling.h"

class ScalingUpConfig {
private:
    Topology* topology;
    int amountOfSamples;
    float remainder;
    float scalingFactor;
    float samplingFraction;
    Sampling* samplingAlgorithm;

    void determineAdditionalSampling();

public:
    ScalingUpConfig(float scalingFactor, float samplingFraction, Topology* topology, Sampling* samplingAlgorithm);

    bool hasSamplingRemainder() const;

    float getRemainder() const;

    void setRemainder(float remainder);

    int getAmountOfSamples() const;

    void setAmountOfSamples(int amountOfSamples);

    float getScalingFactor();

    float getSamplingFraction();

    Topology* getTopology();

    void setTopology(Topology* topology);

    Sampling* getSamplingAlgorithm();
};


#endif //GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H
