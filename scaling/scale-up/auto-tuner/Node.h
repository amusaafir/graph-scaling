//
// Created by Ahmed on 23-7-19.
//

#ifndef GRAPH_SCALING_TOOL_NODE_H
#define GRAPH_SCALING_TOOL_NODE_H

#include "SuggestedParameters.h"

template<typename T>

class Node {
private:
    void printChild(int indent, std::string name, Node* node) {
        std::cout << getTabs(indent) << name << ": " << std::endl;
        if (node != NULL) {
            std::cout << getTabs(indent) << "{" << std::endl;

            node->printPreorderFromCurrentNode(indent + 2);

            std::cout << getTabs(indent) << "}" << std::endl;
        } else {
            std::cout << getTabs(indent) << "--" << "{}" << std::endl;;
        }
    }

    std::string getTabs(int indent) {
        std::string tabs;

        for (int i = 0 ; i < indent; i++) {
            tabs += "-";
        }

        return tabs;
    }

public:
    SuggestedParameters suggestedParameters;
    T value;
    Node<T>* left = NULL;
    Node<T>* right = NULL;
    bool isHeuristic = false;
    float deviance;
    int numberOfHits = 0;

    Node(T value, SuggestedParameters suggestedParameters, bool isHeuristic, float deviance) {
        this->value = value;
        this->suggestedParameters = suggestedParameters;
        this->isHeuristic = isHeuristic;
        this->deviance = deviance;
    }

    void addNode(T val, SuggestedParameters suggestedParameters, bool isHeuristic, float deviance) {
        if (val <= value) {
            if (left == NULL) {
                left = new Node<T>(val, suggestedParameters, isHeuristic, deviance);
            } else {
                left->addNode(val, suggestedParameters, isHeuristic, deviance);
            }
        } else {
            if (this->right == NULL) {
                this->right = new Node<T>(val, suggestedParameters, isHeuristic, deviance);
            } else {
                this->right->addNode(val, suggestedParameters, isHeuristic, deviance);
            }
        }
    }

    void printPreorderFromCurrentNode(int indent = 0) {
        std::cout << getTabs(indent) << "Val: " << value << ", Topology: " << suggestedParameters.topology->getName()
                  << ", Nr. intercon: " << suggestedParameters.topology->getBridge()->getNumberOfInterconnections()
                  << ", Heuristic: " << isHeuristic << ", Deviance: " << deviance * 100 << "%" << std::endl;

        printChild(indent, "L", left);
        printChild(indent, "R", right);
    }
};


#endif //GRAPH_SCALING_TOOL_NODE_H
