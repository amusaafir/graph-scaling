//
// Created by Ahmed on 25-1-18.
//

#ifndef GRAPH_SCALING_TOOL_USERINPUTPROMPT_H
#define GRAPH_SCALING_TOOL_USERINPUTPROMPT_H


#include "UserInput.h"
#include "../../scaling/scale-up/bridge/RandomBridge.h"

class UserInputPrompt : public UserInput {
public:

    UserInputPrompt();

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


#endif //GRAPH_SCALING_TOOL_USERINPUTPROMPT_H
