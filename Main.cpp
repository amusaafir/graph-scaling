#include <iostream>
#include "Scaler.h"

std::string logo = "  __ ___  __  ___ _  _    __   ___ __  _   _ __  _  __   _____ __   __  _    \n"
        " / _] _ \\/  \\| _,\\ || | /' _/ / _//  \\| | | |  \\| |/ _] |_   _/__\\ /__\\| |   \n"
        "| [/\\ v / /\\ | v_/ >< | `._`.| \\_| /\\ | |_| | | ' | [/\\   | || \\/ | \\/ | |_  \n"
        " \\__/_|_\\_||_|_| |_||_| |___/ \\__/_||_|___|_|_|\\__|\\__/   |_| \\__/ \\__/|___| ";

std::string version = "v1.0";

int main() {
    std::cout << logo << version << std::endl;

    Scaler* scaler = new Scaler();
    delete(scaler);

    return 0;
}