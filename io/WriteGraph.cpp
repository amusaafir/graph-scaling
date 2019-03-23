//
// Created by Ahmed on 23-3-19.
//

#include "WriteGraph.h"

void WriteGraph::write() {
    std::string outputPath = outputFolderPath + "/" + getFileName();

    std::cout << "Writing output file to " << outputPath << std::endl;

    this->writeToFile();

    std::cout << "Finished writing output file." << std::endl;
}