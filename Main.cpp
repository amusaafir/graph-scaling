#include <iostream>
#include "scaling/ScalingManager.h"
#include "loader/GraphLoader.h"

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

    ScalingManager* scalingManager = new ScalingManager(graph);
    scalingManager->scaleUp(3.7, 0.5);
    //scalingManager->scaleDown(0.5);
    delete(scalingManager);

    delete(graph);

    return 0;
}