//
// Created by Ahmed on 25-1-18.
//

#ifndef GRAPH_SCALING_TOOL_USERINPUTCMD_H
#define GRAPH_SCALING_TOOL_USERINPUTCMD_H


#include "UserInput.h"
#include "../../scaling/scale-up/bridge/RandomBridge.h"
#include "../../scaling/scale-down/Sampling.h"
#include <sstream>
#include <array>
#include <string>

class UserInputCMD : public UserInput {
private:
    std::map<char, std::string> inputArguments;
    void insertArgumentValues(int argc, char* argv[]);

public:
    UserInputCMD(int argc, char* argv[]);

    std::string getInputGraphPath();

    int getScalingType();

    std::string getOutputGraphPath();

    Bridge *getBridge();

    bool addDirectedBridges();

    long long getNumberOfInterconnections();

    Topology *getTopology();

    float getSamplingFraction();

    float getScalingFactor();

    Sampling* getSamplingAlgorithm(Graph* graph);
};


#endif //GRAPH_SCALING_TOOL_USERINPUTCMD_H
