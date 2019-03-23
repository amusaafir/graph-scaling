//
// Created by Ahmed on 23-3-19.
//

#ifndef GRAPH_SCALING_TOOL_WRITEGRAPH_H
#define GRAPH_SCALING_TOOL_WRITEGRAPH_H

#include <iostream>
#include "../graph/Graph.h"

class WriteGraph {
protected:
    const std::string OUTPUT_EXTENSION = ".csv";
    std::string outputFolderPath;

    virtual void writeToFile() = 0;

    virtual std::string getFileName() = 0;

public:
    void write();
};


#endif //GRAPH_SCALING_TOOL_WRITEGRAPH_H
