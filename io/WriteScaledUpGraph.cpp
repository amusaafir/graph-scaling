//
// Created by Ahmed on 5-1-18.
//

#include "WriteScaledUpGraph.h"

WriteScaledUpGraph::WriteScaledUpGraph(std::string outputFolderPath, std::vector<Graph*> samples, std::vector<Edge<std::string>*> bridges) {
    this->outputFolderPath = outputFolderPath;
    this->samples = samples;
    this->bridges = bridges;
}

void WriteScaledUpGraph::writeToFile(ScaleUpSamplesInfo* scaleUpSamplesInfo) {
   // std::ofstream fileInfo("expanded_graph_" + scaleUpSamplesInfo->getTopology()->);

}