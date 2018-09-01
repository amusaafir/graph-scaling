//
// Created by Ahmed on 14-11-17.
//

#include "Sampling.h"
#include "../../io/WriteSampledGraph.h"

Sampling::Sampling(Graph* graph, std::string samplingAlgorithmName) {
    this->graph = graph;
    std::cout << "Selected " << samplingAlgorithmName << " as sampling algorithm." << std::endl;
}

long long Sampling::getNumberOfVerticesFromFraction(float fraction) {
    return this->graph->getVertices().size() * fraction;
}

long long Sampling::getNumberOfEdgesFromFraction(float fraction) {
    return this->graph->getEdges().size() * fraction;
}

long long Sampling::getRandomIntBetweenRange(long long min, long long max) {
    std::mt19937 engine(seed());
    std::uniform_int_distribution<long long> dist(min, max);

    return dist(engine);
}

void Sampling::run(float fraction, std::string outputPath) {
    Graph* graph = sample(fraction);

    WriteSampledGraph* writeSampledGraph = new WriteSampledGraph(graph, outputPath, fraction);
    writeSampledGraph->writeToFile();
}
