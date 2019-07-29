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
    std::cout << "Tuning diameter:" << std::endl;

    if (diameterRoot == NULL) {
        // Build initial diameter tree
        std::cout << "Building initial diameter tree.." << std::endl;

        for (int i = 0; i < sizeof(topologies) / sizeof(topologies[0]); i++) {
            SuggestedParameters suggestedParameters;
            suggestedParameters.topology = topologies[i]->createTopology(new RandomBridge(1, false));

            if (diameterRoot == NULL) {
                diameterRoot = new Node<int>(topologies[i]->getMaxDiameter(), suggestedParameters);
            } else {
                diameterRoot->addNode(topologies[i]->getMaxDiameter(), suggestedParameters);
            }
        }

        std::cout << "Finished building diameter tree." << std::endl;
    }

    // Check if a given new node is not empty and

    diameterRoot->printPreorderFromCurrentNode();

    return findClosestMatch();
}

void Autotuner::addNodeToDiameterTree(int diameter, SuggestedParameters suggestedParameters) {
    std::cout << "Adding node to diameter tree.." << std::endl;

    if (diameterRoot == NULL) {
        diameterRoot = new Node<int>(diameter, suggestedParameters);
    } else {
        diameterRoot->addNode(diameter, suggestedParameters);
    }

    // Check if a given new node is not empty and

    std::cout << "Current diameter tree: " << std::endl;

    diameterRoot->printPreorderFromCurrentNode();
}


SuggestedParameters Autotuner::findClosestMatch() {
    SuggestedParameters currentClosestDiameter;


}

bool Autotuner::isInsideDiameterMargin(int currentDiameter) {
    const float MARGIN = 0.2;
    const float real = originalDiameter / currentDiameter;

    return real <= MARGIN;
}