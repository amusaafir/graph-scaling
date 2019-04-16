//
// Created by Ahmed on 5-1-18.
//

#include "WriteScaledUpGraph.h"

WriteScaledUpGraph::WriteScaledUpGraph(std::string outputFolderPath, std::vector<Graph*> samples,
                                       std::vector<Edge<std::string>> bridges, ScalingUpConfig* scalingUpConfig) {
    this->outputFolderPath = outputFolderPath;
    this->samples = samples;
    this->bridges = bridges;
    this->scalingUpConfig = scalingUpConfig;
}

void WriteScaledUpGraph::writeToFile() {
    std::ofstream outputFile(outputFolderPath + "/" + getFileName());

    if (outputFile.is_open()) {
        writeEdgesFromSamples(outputFile);
        writeEdgesFromBridges(outputFile);
        outputFile.close();
    }
}

std::string WriteScaledUpGraph::getFileName() {
    std::stringstream samplingFractionStream, scalingFactorStream;
    samplingFractionStream << std::fixed << std::setprecision(2) << this->scalingUpConfig->getSamplingFraction();
    scalingFactorStream << std::fixed << std::setprecision(2) << this->scalingUpConfig->getScalingFactor();

    std::string samplingFractionFormat = samplingFractionStream.str();
    std::string scalingFactorFormat = scalingFactorStream.str();

    samplingFractionFormat = samplingFractionFormat.replace(samplingFractionFormat.find("."), 1, "-");
    scalingFactorFormat = scalingFactorFormat.replace(scalingFactorFormat.find("."), 1, "-");

    std::string filename = "up_" + scalingUpConfig->getSamplingAlgorithm()->getSamplingAlgorithmName() + "_"
                                    + samplingFractionFormat + "_"
                                    + scalingFactorFormat + "_"
                                    + this->scalingUpConfig->getTopology()->getName() + "_"
                                    + std::to_string(this->scalingUpConfig->getTopology()->getBridge()->getNumberOfInterconnections())
                                    + this->scalingUpConfig->getTopology()->getBridge()->getName();

    return filename + OUTPUT_EXTENSION;
}

void WriteScaledUpGraph::writeEdgesFromBridges(std::ofstream &outputFile) const {
    for (long long i = 0; i < bridges.size(); i++) {
            Edge<std::string> edge = bridges[i];
            outputFile << edge.getSource() + "\t" + edge.getTarget() + "\n";
    }
}

void WriteScaledUpGraph::writeEdgesFromSamples(std::ofstream &outputFile) const {
    for (long long i =0; i < samples.size(); i++) {
        std::string graphIdentifier = samples[i]->getIdentifier();

        for(long long p = 0; p < samples[i]->getEdges().size(); p++) {
            Edge<long long> edge = samples[i]->getEdges()[p];
            outputFile << std::to_string(edge.getSource()) + graphIdentifier + "\t" + std::to_string(edge.getTarget()) + graphIdentifier + "\n";
        }
    }
}