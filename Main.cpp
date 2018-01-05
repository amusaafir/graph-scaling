#include <iostream>
#include "scaling/Scaling.h"
#include "loader/GraphLoader.h"
#include "scaling/scale-up/IdentifierTracker.h"
#include "scaling/scale-up/bridge/RandomBridge.h"
#include "scaling/scale-up/topology/StarTopology.h"


std::string logo = "  __ ___  __  ___ _  _    __   ___ __  _   _ __  _  __   _____ __   __  _    \n"
        " / _] _ \\/  \\| _,\\ || | /' _/ / _//  \\| | | |  \\| |/ _] |_   _/__\\ /__\\| |   \n"
        "| [/\\ v / /\\ | v_/ >< | `._`.| \\_| /\\ | |_| | | ' | [/\\   | || \\/ | \\/ | |_  \n"
        " \\__/_|_\\_||_|_| |_||_| |___/ \\__/_||_|___|_|_|\\__|\\__/   |_| \\__/ \\__/|___| ";

std::string version = "v1.0";


int main() {
    std::cout << logo << version << std::endl;

    GraphLoader* graphLoader = new GraphLoader();
    Graph* graph = graphLoader->loadGraph("/home/aj/Documents/graph_datasets/facebook_combined.txt");

    delete(graphLoader);

    Scaling* scaling = new Scaling(graph);
    scaling->scaleUp(new ScaleUpSamplesInfo(new StarTopology(new RandomBridge(100, false)), 3.7, 0.5));
    //scaling->scaleDown(0.5);
    delete(scaling);

    delete(graph);

    return 0;
}