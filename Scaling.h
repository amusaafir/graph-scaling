//
// Created by Ahmed on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SCALER_H
#define GRAPH_SCALING_TOOL_SCALER_H

#include "loader/GraphLoader.h"

class Scaling {
private:
    GraphLoader* graphLoader;
    void initScaling();

public:
    Scaling();
    ~Scaling();
};


#endif //GRAPH_SCALING_TOOL_SCALER_H
