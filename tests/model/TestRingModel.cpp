//
// Created by Ahmed on 23-7-19.
//

#include <gtest/gtest.h>
#include "../../scaling/scale-up/auto-tuner/model/ChainModel.h"

TEST(ChainModel, testMaxDiameter) {
ChainModel chainModel(8, 6, 3);

ASSERT_EQ(chainModel.getMaxDiameter(), 53);
}
