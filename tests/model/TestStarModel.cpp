//
// Created by Ahmed on 23-7-19.
//

#include <gtest/gtest.h>
#include "../../scaling/scale-up/auto-tuner/model/StarModel.h"

TEST(StarModel, testMaxDiameter) {
    StarModel starModel(8, 6, 3);

    ASSERT_EQ(starModel.getMaxDiameter(), 26);
}