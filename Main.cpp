#include <iostream>
#include <map>
#include "scaling/Scaling.h"
#include "io/GraphLoader.h"
#include "scaling/scale-up/IdentifierTracker.h"
#include "scaling/scale-up/bridge/RandomBridge.h"
#include "scaling/scale-up/topology/StarTopology.h"
#include "scaling/scale-up/topology/ChainTopology.h"
#include "scaling/scale-up/topology/RingTopology.h"
#include "scaling/scale-up/topology/FullyConnectedTopology.h"


std::string logo = "  ____                        _        ____                        \n"
        " / ___|  __ _ _ __ ___  _ __ | | ___  / ___| _ __  _ __ __ _ _   _ \n"
        " \\___ \\ / _` | '_ ` _ \\| '_ \\| |/ _ \\ \\___ \\| '_ \\| '__/ _` | | | |\n"
        "  ___) | (_| | | | | | | |_) | |  __/  ___) | |_) | | | (_| | |_| |\n"
        " |____/ \\__,_|_| |_| |_| .__/|_|\\___| |____/| .__/|_|  \\__,_|\\__, |\n"
        "                       |_|                  |_|              |___/ ";

std::string version = "v1.0";


int main() {
    std::cout << logo << version << std::endl;

    std::cout << "Note: the input and output paths may not contain any white space (newlines/spaces)." << std::endl;

    // Input graph
    std::string specInputGraphPath;
    std::cout << "Enter path input graph:" << std::endl;
    std::cin >> specInputGraphPath;
    //"/home/aj/Documents/graph_datasets/graph.txt"

    // Output folder
    std::string specOutputFolder;
    std::cout << "Enter path output folder:" << std::endl;
    std::cin >> specOutputFolder;
    //"/home/aj/Documents/output_scaling"

    // TODO: High degree bridging
    Bridge* bridge = new RandomBridge(2, false);

    // Topology
    char specTopology;
    std::cout << "Topology: [s]tar; [c]hain; [r]ing; [f]ullyconnected:" << std::endl;
    std::cin >> specTopology;

    std::map<char, Topology*> topologies;
    topologies['s'] = new StarTopology(bridge);
    topologies['c'] = new ChainTopology(bridge);
    topologies['r'] = new RingTopology(bridge);
    topologies['f'] = new FullyConnectedTopology(bridge);
    Topology* topology = topologies[specTopology];

    // Scaling factor
    float scalingFactor;
    std::cout << "Scaling factor: " << std::endl;
    std::cin >> scalingFactor;

    // Sampling fraction
    float samplingFraction;
    std::cout << "Sample size per graph: " << std::endl;
    std::cin >> samplingFraction;

    GraphLoader* graphLoader = new GraphLoader();
    Graph* graph = graphLoader->loadGraph(specInputGraphPath);
    delete(graphLoader);

    Scaling* scaling = new Scaling(graph);
    ScaleUpSamplesInfo* scaleUpSamplesInfo = new ScaleUpSamplesInfo(topology, scalingFactor, samplingFraction);
    scaling->scaleUp(scaleUpSamplesInfo, specOutputFolder);
    //scaling->scaleDown(0.5);

    delete(scaleUpSamplesInfo);
    delete(topologies['s']);
    delete(topologies['c']);
    delete(topologies['r']);
    delete(topologies['f']);
    delete(bridge);
    delete(graph);
    delete(scaling);

    return 0;
}