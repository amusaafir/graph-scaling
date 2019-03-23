//
// Created by Ahmed on 26-1-18.
//

#include "WriteSampledGraph.h"

WriteSampledGraph::WriteSampledGraph(Graph* graph, std::string outputFolderPath, float fraction) {
    this->graph = graph;
    this->outputFolderPath = outputFolderPath;
    this->fraction = fraction;
}

void WriteSampledGraph::writeToFile() {
    std::ofstream outputFile(outputFolderPath + "/" + getFileName());

    if (outputFile.is_open()) {
        writeGraphEdges(outputFile);
        outputFile.close();
    }
}

std::string WriteSampledGraph::getFileName() {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << fraction;

    return "sampled_graph_" + stream.str() + OUTPUT_EXTENSION;
}


void WriteSampledGraph::writeGraphEdges(std::ofstream &outputFile) {
    for(long long i = 0; i < graph->getEdges().size(); i++) {
        Edge<long long> edge = graph->getEdges()[i];
        outputFile << std::to_string(edge.getSource()) + "\t" + std::to_string(edge.getTarget()) + "\n";
    }
}