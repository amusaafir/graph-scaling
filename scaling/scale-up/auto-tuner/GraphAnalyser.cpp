//
// Created by Ahmed on 30-3-19.
//

#include "GraphAnalyser.h"

GraphAnalyser::GraphAnalyser() {
}

void GraphAnalyser::loadGraph(std::vector<Graph*> samples, std::vector<Edge<std::string>> bridges) {
    if (!graph.Empty()) {
        graph.Clr();
    }

    graph = TUNGraph::New();
    int coordinate = 0;
    std::unordered_map<std::string, int> map_from_edge_to_coordinate;

    for (long long i =0; i < samples.size(); i++) {
        std::string graphIdentifier = samples[i]->getIdentifier();

        for(long long p = 0; p < samples[i]->getEdges().size(); p++) {
            Edge<long long> edge = samples[i]->getEdges()[p];

            int iSource;
            int iTarget;
            std::string source = std::to_string(edge.getSource()) + graphIdentifier;
            std::string target = std::to_string(edge.getTarget()) + graphIdentifier;

            if (map_from_edge_to_coordinate.count(source)) {
                iSource = map_from_edge_to_coordinate[source];
            } else {
                map_from_edge_to_coordinate[source] = coordinate;

                iSource = coordinate;
                coordinate++;
            }

            if (map_from_edge_to_coordinate.count(target)) {
                iTarget = map_from_edge_to_coordinate[target];
            } else {
                map_from_edge_to_coordinate[target] = coordinate;

                iTarget = coordinate;
                coordinate++;
            }

            graph->AddEdge2(iSource, iTarget);
        }
    }

    for (long long i = 0; i < bridges.size(); i++) {
        Edge<std::string> edge = bridges[i];

        int iSource;
        int iTarget;

        if (map_from_edge_to_coordinate.count(edge.getSource())) {
            iSource = map_from_edge_to_coordinate[edge.getSource()];
        } else {
            map_from_edge_to_coordinate[edge.getSource()] = coordinate;

            iSource = coordinate;
            coordinate++;
        }

        if (map_from_edge_to_coordinate.count(edge.getTarget())) {
            iTarget = map_from_edge_to_coordinate[edge.getTarget()];
        } else {
            map_from_edge_to_coordinate[edge.getTarget()] = coordinate;

            iTarget = coordinate;
            coordinate++;
        }


        graph->AddEdge2(iSource, iTarget);
    }

    std::cout  << "Number of nodes in scaled up graph: " << graph->GetNodes() << std::endl;
}


int GraphAnalyser::calculateDiameter() {
    return TSnap::GetBfsFullDiam(graph, DIAMETER_TEST_NODES);
}

bool GraphAnalyser::deleteGraph() {
    graph->Clr();

    return graph->Empty();
}