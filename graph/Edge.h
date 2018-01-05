//
// Created by Ahmed on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_EDGE_H
#define GRAPH_SCALING_TOOL_EDGE_H

template<typename T>

class Edge {
private:
    T source;
    T target;
public:
    Edge(T source, T target) {
        this->source = source;
        this->target = target;
    }

    T getSource() const {
        return source;
    }

    void setSource(T source) {
        this->source = source;
    }

    T getTarget() const {
        return target;
    }

    void setTarget(T target) {
        this->target = target;
    }
};


#endif //GRAPH_SCALING_TOOL_EDGE_H
