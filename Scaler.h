//
// Created by root on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALER_H
#define GRAPH_SCALING_TOOL_SCALER_H

#include "loader/GraphLoader.h"

class Scaler {
private:
    GraphLoader* graphLoader;
    void initScaling();

public:
    Scaler();
    ~Scaler();
};


#endif //GRAPH_SCALING_TOOL_SCALER_H
