//
// Created by Ahmed on 23-7-19.
//

#include <gtest/gtest.h>
#include "../../scaling/scale-up/auto-tuner/model/RingModel.h"

TEST(RingModel, testMaxDiameter) {
    RingModel ringModel(8, 6, 3);

ASSERT_EQ(ringModel.getMaxDiameter(), 54);
}
