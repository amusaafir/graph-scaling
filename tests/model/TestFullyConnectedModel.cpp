//
// Created by Ahmed on 23-7-19.
//

#include <gtest/gtest.h>
#include "../../scaling/scale-up/auto-tuner/model/FullyConnectedModel.h"

TEST(FullyConnectedModel, testMaxDiameter) {
    FullyConnectedModel fullyConnectedModel(8, 6, 3);

    ASSERT_EQ(fullyConnectedModel.getMaxDiameter(), 17);
}
