//
// Created by Ahmed on 20-11-17.
//

#ifndef GRAPH_SCALING_TOOL_IDENTIFIERTRACKER_H
#define GRAPH_SCALING_TOOL_IDENTIFIERTRACKER_H

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

class IdentifierTracker {
private:
    const int MAX_GRAPH_SAMPLE_INDEX = 26 * 26;
    int currentGraphSampleIndex;

public:
    IdentifierTracker();
    std::string createNewIdentifier();
};


#endif //GRAPH_SCALING_TOOL_IDENTIFIERTRACKER_H
