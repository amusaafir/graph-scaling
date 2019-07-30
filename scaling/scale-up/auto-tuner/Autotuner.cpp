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
#include <limits>
#include <cmath>

Autotuner::Autotuner(int ORIGINAL_DIAMETER, int targetDiameter, int numberOfSamples) {
    // Build diameter binary tree
    bridge = new RandomBridge(1, false);

    const float scalingFactor = 3.0f;

    this->targetDiameter = targetDiameter;

    topologies[0] = new StarModel(ORIGINAL_DIAMETER, numberOfSamples, scalingFactor);
    topologies[1] = new ChainModel(ORIGINAL_DIAMETER, numberOfSamples, scalingFactor);
    topologies[2] = new RingModel(ORIGINAL_DIAMETER, numberOfSamples, scalingFactor);
    topologies[3] = new FullyConnectedModel(ORIGINAL_DIAMETER, numberOfSamples, scalingFactor);
}

void tuneDiameter() {
    /*std::cout << "Tuning diameter:" << std::endl;

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

     */
}

void Autotuner::addNodeToDiameterTree(int diameter, SuggestedParameters suggestedParameters, bool isHeuristic) {
    std::cout << "Adding node to diameter tree.." << std::endl;

    float deviance = 1 - computeDeviance(diameter, targetDiameter);

    if (diameterRoot == NULL) {
        diameterRoot = new Node<int>(diameter, suggestedParameters, isHeuristic, deviance);
    } else {
        diameterRoot->addNode(diameter, suggestedParameters, isHeuristic,  deviance);
    }

    // Check if a given new node is not empty and

    std::cout << "Current diameter tree: " << std::endl;

    diameterRoot->printPreorderFromCurrentNode();
}

float Autotuner::computeDeviance(int diameter, int targetDiameter)  {
    return ((float) diameter / (float) targetDiameter);
}

SuggestedParameters Autotuner::getNewSuggestion() {
    SuggestedParameters suggestedParams = findClosestDiameterNode(diameterRoot)->suggestedParameters;

    std::cout<<"Closest current suggestion: " << suggestedParams.getParameterStringRepresentation() << std::endl;

    return suggestedParams;
}

Node<int>* Autotuner::findClosestDiameterNode(Node<int>* node) {
    if (node == NULL) {
        return NULL;
    }

    if (currentClosestDiameterNode == NULL) {
        currentClosestDiameterNode = node;
    } else {
        if (!node->isHeuristic && std::abs(computeDeviance(node->value, targetDiameter)) <= std::abs(computeDeviance(currentClosestDiameterNode->value, targetDiameter))) {
            currentClosestDiameterNode = node;
        }
    }

    findClosestDiameterNode(node->left);
    findClosestDiameterNode(node->right);

    return currentClosestDiameterNode;
}

bool Autotuner::isInsideDiameterMargin(int currentDiameter) {
    const float MARGIN = 0.2;
    const float real = originalDiameter / currentDiameter;

    return real <= MARGIN;
}