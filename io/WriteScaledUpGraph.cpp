//
// Created by Ahmed on 5-1-18.
//

#include "WriteScaledUpGraph.h"

WriteScaledUpGraph::WriteScaledUpGraph(std::string outputFolderPath, std::vector<Graph*> samples, std::vector<Edge<std::string>> bridges) {
    this->outputFolderPath = outputFolderPath;
    this->samples = samples;
    this->bridges = bridges;
}

void WriteScaledUpGraph::writeToFile(ScaleUpSamplesInfo* scaleUpSamplesInfo) {
    std::string filename = "expanded_graph_"
                           + std::to_string(scaleUpSamplesInfo->getSamplingFraction()) + "_"
                           + std::to_string(scaleUpSamplesInfo->getScalingFactor()) + "_"
                           + scaleUpSamplesInfo->getTopology()->getName() + "_"
                           + scaleUpSamplesInfo->getTopology()->getBridge()->getName();


    std::cout << "Writing output file to " << outputFolderPath + "/" + filename + ".txt" << std::endl;

    std::ofstream outputFile(outputFolderPath + "/" + filename + ".txt");

    if (outputFile.is_open()) {
        writeEdgesFromSamples(outputFile);
        writeEdgesFromBridges(outputFile);
        outputFile.close();
    }

    std::cout << "Finished writing output file." << std::endl;
}

void WriteScaledUpGraph::writeEdgesFromBridges(std::ofstream &outputFile) const {
    for (int i = 0; i < bridges.size(); i++) {
            Edge<std::__cxx11::string> edge = bridges[i];
            outputFile << edge.getSource() + ", " + edge.getTarget() + "\n";
        }
}

void WriteScaledUpGraph::writeEdgesFromSamples(std::ofstream &outputFile) const {
    for (int i =0; i < samples.size(); i++) {
        std::string graphIdentifier = samples[i]->getIdentifier();

        for(int p = 0; p < samples[i]->getEdges().size(); p++) {
            Edge<int> edge = samples[i]->getEdges()[p];
            outputFile << std::to_string(edge.getSource()) + graphIdentifier + ", " + std::to_string(edge.getTarget()) + graphIdentifier + "\n";
        }
    }
}