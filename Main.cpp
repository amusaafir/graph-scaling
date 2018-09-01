#include <iostream>
#include <map>
#include "scaling/Scaling.h"
#include "io/GraphLoader.h"
#include "io/user-input/UserInputPrompt.h"
#include "io/user-input/UserInputCMD.h"

std::string logo = "  ____                        _        ____                        \n"
        " / ___|  __ _ _ __ ___  _ __ | | ___  / ___| _ __  _ __ __ _ _   _ \n"
        " \\___ \\ / _` | '_ ` _ \\| '_ \\| |/ _ \\ \\___ \\| '_ \\| '__/ _` | | | |\n"
        "  ___) | (_| | | | | | | |_) | |  __/  ___) | |_) | | | (_| | |_| |\n"
        " |____/ \\__,_|_| |_| |_| .__/|_|\\___| |____/| .__/|_|  \\__,_|\\__, |\n"
        "                       |_|                  |_|              |___/ ";

void scaleUp(Graph *graph, UserInput *userInput);

void scaleDown(Graph* graph, UserInput *userInput);

UserInput* getUserInput(int argc, char* argv[]);

void scaleGraph(UserInput *userInput, Graph *graph);

int main(int argc, char* argv[]) {
    std::cout << logo << std::endl;

    UserInput* userInput = getUserInput(argc, argv);
    GraphLoader* graphLoader = new GraphLoader();
    Graph* graph = graphLoader->loadGraph(userInput->getInputGraphPath());

    scaleGraph(userInput, graph);

    delete(userInput);
    delete(graphLoader);

    return 0;
}

void scaleGraph(UserInput* userInput, Graph *graph) {
    Scaling* scaling = new Scaling(graph, userInput);

    if (userInput->getScalingType() == 1) {
        scaling->scaleUp();
    } else {
        scaling->scaleDown();
    }

    delete(scaling);
}

UserInput* getUserInput(int argc, char* argv[]) {
    if (argc > 1) {
        return new UserInputCMD(argc, argv);
    }

    return new UserInputPrompt();
}