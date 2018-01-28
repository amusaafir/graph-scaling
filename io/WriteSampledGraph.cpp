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
    std::string outputPath = outputFolderPath + "/" + createFilename() + ".txt";

    std::cout << "Writing output file to " << outputPath << std::endl;

    std::ofstream outputFile(outputPath);

    if (outputFile.is_open()) {
        writeGraphEdges(outputFile);
        outputFile.close();
    }

    std::cout << "Finished writing output file." << std::endl;
}

std::string WriteSampledGraph::createFilename() {
    return "sampled_graph_" + std::to_string(fraction);
}


void WriteSampledGraph::writeGraphEdges(std::ofstream &outputFile) {
    for(int p = 0; p < graph->getEdges().size(); p++) {
        Edge<int> edge = graph->getEdges()[p];
        outputFile << std::to_string(edge.getSource()) + "\t" + std::to_string(edge.getTarget()) + "\n";
    }
}