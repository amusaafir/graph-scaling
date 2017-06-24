#pragma once

#include "BridgeSelection.h"
#include "../SampledGrapHVersionStruct.h"

class RandomBridge : public BridgeSelection {
public:
	int RandomBridge::select_bridges(Sampled_Graph_Version* graph);
};