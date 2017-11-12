//
// Created by Ahmed on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_EDGE_H
#define GRAPH_SCALING_TOOL_EDGE_H

class Edge {
private:
    int source;
    int target;
public:
    Edge(int source, int target);

    int getSource() const;

    void setSource(int source);

    int getTarget() const;

    void setTarget(int target);
};


#endif //GRAPH_SCALING_TOOL_EDGE_H
