//
// Created by Ahmed on 12-12-17.
//

#include "Bridge.h"

Bridge::Bridge(Graph *left, Graph *right, int numberOfInterconnections, bool forceUndirectedEdges) {
        this->left = left;
        this->right = right;
        this->numberOfInterconnections = numberOfInterconnections;
        this->forceUndirectedEdges = forceUndirectedEdges;
}