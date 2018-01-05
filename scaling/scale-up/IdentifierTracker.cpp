//
// Created by Ahmed on 20-11-17.
//

#include "IdentifierTracker.h"

IdentifierTracker::IdentifierTracker() {
    this->currentGraphSampleIndex = 1;
}

std::string IdentifierTracker::createNewIdentifier() {
    char firstIdentifier = 'a', secondIdentifier = 'a';

    firstIdentifier += fmod(currentGraphSampleIndex, 26);
    secondIdentifier += (currentGraphSampleIndex / 26);

    currentGraphSampleIndex++;

    std::string identifier;
    identifier += firstIdentifier;
    identifier += secondIdentifier;

    return identifier;
}