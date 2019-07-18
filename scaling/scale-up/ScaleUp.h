//
// Created by Ahmed on 15-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALEUP_H
#define GRAPH_SCALING_TOOL_SCALEUP_H

#include <math.h>
#include "ScalingUpConfig.h"
#include "../../graph/Graph.h"
#include "../scale-down/Sampling.h"
#include "IdentifierTracker.h"
#include "../../io/WriteScaledUpGraph.h"
#include "auto-tuner/GraphAnalyser.h"


class ScaleUp {
private:
    Graph* graph;
    ScalingUpConfig* scaleUpSamplesInfo;
    std::string outputFolder;

    /**
     * Checks whether an additional sampling operation should be executed (if there is a remainder left).
     * @param scaleUpSamplesInfo
     * @param currentLoopIteration
     * @return
     */
    bool shouldSampleRemainder(ScalingUpConfig *scaleUpSamplesInfo, int currentLoopIteration);

    /**
     * Print scale up configuration.
     */
    void printScaleUpSetup();

    /**
     * Creates different samples based on the given scale up configuration
     * @return list of sampled graphs.
     */
    std::vector<Graph*> createDistinctSamples();

    /**
     * Creates a single graph sample and puts it inside the samples vector.
     * @param samples - vector to put the newly created graph sample in.
     * @param samplingFraction - sample size
     * @param identifier - (unique) id to keep track of the vertices.
     */
    void createSample(std::vector<Graph*> &samples, float samplingFraction, std::string identifier);

public:
    ScaleUp(Graph* graph, ScalingUpConfig* scaleUpSamplesInfo, std::string outputFolder);

    /**
     * Executes a scale-up operation and writes the result in the given output folder.
     */
    void run();
};


#endif //GRAPH_SCALING_TOOL_SCALEUP_H
