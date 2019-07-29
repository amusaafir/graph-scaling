//
// Created by Ahmed on 30-3-19.
//

#include <map>
#include "Autotuner.h"
#include "../topology/StarTopology.h"
#include "../topology/ChainTopology.h"
#include "../topology/RingTopology.h"
#include "../topology/FullyConnectedTopology.h"
#include "../bridge/RandomBridge.h"
#include "model/StarModel.h"
#include "model/ChainModel.h"
#include "model/RingModel.h"
#include "model/FullyConnectedModel.h"

Autotuner::Autotuner(int originalDiameter, int numberOfSamples) {
    // Build diameter binary tree
    bridge = new RandomBridge(1, false);

    const float scalingFactor = 3.0f;

    topologies[0] = new StarModel(originalDiameter, numberOfSamples, scalingFactor);
    topologies[1] = new ChainModel(originalDiameter, numberOfSamples, scalingFactor);
    topologies[2] = new RingModel(originalDiameter, numberOfSamples, scalingFactor);
    topologies[3] = new FullyConnectedModel(originalDiameter, numberOfSamples, scalingFactor);
}

SuggestedParameters Autotuner::tuneDiameter() {
    // Build initial diameter tree
    for (int i = 0; i < sizeof(topologies) / sizeof(topologies[0]); i++) {
        SuggestedParameters suggestedParameters;
        suggestedParameters.topology = topologies[i]->createTopology(new RandomBridge(1, false));

        if (diameterRoot == NULL) {
            diameterRoot = new Node<int>(topologies[i]->getMaxDiameter(), suggestedParameters);
        } else {
            diameterRoot->addNode(topologies[i]->getMaxDiameter(), suggestedParameters);
        }
    }

    diameterRoot->printPreorderFromCurrentNode();
}