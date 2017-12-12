//
// Created by Ahmed on 16-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H
#define GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H

#include <math.h>

class ScaleUpSamplesInfo {
private:
    int amountOfSamples;
    float remainder;

    void determineAdditionalSampling();

public:
    ScaleUpSamplesInfo(float scalingFactor, float samplingFraction);

    bool hasSamplingRemainder() const;

    float getRemainder() const;

    void setRemainder(float remainder);

    int getAmountOfSamples() const;

    void setAmountOfSamples(int amountOfSamples);
};


#endif //GRAPH_SCALING_TOOL_SCALEUPSAMPLEREMAINDER_H
