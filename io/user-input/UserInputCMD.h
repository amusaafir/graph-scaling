//
// Created by Ahmed on 25-1-18.
//

#ifndef GRAPH_SCALING_TOOL_USERINPUTCMD_H
#define GRAPH_SCALING_TOOL_USERINPUTCMD_H


#include "UserInput.h"
#include "../../scaling/scale-up/bridge/RandomBridge.h"
#include <sstream>
#include <array>

class UserInputCMD : public UserInput {
private:
    std::map<char, std::string> inputArguments;
    bool isScaleUp = false;
    void insertArgumentValues(int argc, char* argv[]);

public:
    UserInputCMD(int argc, char* argv[]);

    std::string getInputGraphPath();

    int getScalingType();

    std::string getOutputGraphPath();

    Bridge *getBridge();

    bool addDirectedBridges();

    int getNumberOfInterconnections();

    Topology *getTopology();

    float getSamplingFraction();

    float getScalingFactor();
};


#endif //GRAPH_SCALING_TOOL_USERINPUTCMD_H
