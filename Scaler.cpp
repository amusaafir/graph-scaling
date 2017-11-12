//
// Created by root on 12-11-17.
//

#include "Scaler.h"

Scaler::Scaler() {
    initScaling();
}

Scaler::~Scaler() {
    delete(graphLoader);
}

void Scaler::initScaling() {
    graphLoader->loadGraph("/home/aj/Documents/graph_datasets/facebook_combined.txt");
}
