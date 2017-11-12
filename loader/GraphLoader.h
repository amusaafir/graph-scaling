//
// Created by Ahmed on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_GRAPHLOADER_H
#define GRAPH_SCALING_TOOL_GRAPHLOADER_H

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "../graph/Graph.h"

class GraphLoader {
public:
    Graph* loadGraph(std::string path);
};


#endif //GRAPH_SCALING_TOOL_GRAPHLOADER_H
