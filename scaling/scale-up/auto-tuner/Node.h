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

    Node(T value, SuggestedParameters suggestedParameters) {
        this->value = value;
        this->suggestedParameters = suggestedParameters;
    }

    void addNode(T val, SuggestedParameters suggestedParameters) {
        if (val < value) {
            if (left == NULL) {
                left = new Node<T>(val, suggestedParameters);
            } else {
                left->addNode(val, suggestedParameters);
            }
        } else {
            if (this->right == NULL) {
                this->right = new Node<T>(val, suggestedParameters);
            } else {
                this->right->addNode(val, suggestedParameters);
            }
        }
    }

    void printPreorderFromCurrentNode(int indent = 0) {
        std::cout << getTabs(indent) << "Val: " << value << ", Topology: " << suggestedParameters.topology->getName() << std::endl;

        printChild(indent, "L", left);
        printChild(indent, "R", right);
    }
};


#endif //GRAPH_SCALING_TOOL_NODE_H
