//
// Created by Ahmed on 29-3-19.
//

#include <gtest/gtest.h>
#include "../scaling/scale-up/bridge/HighDegreeBridge.h"

TEST(HighDegreeBridge, justATest) {
    HighDegreeBridge* hdb = new HighDegreeBridge(10, false);

    ASSERT_EQ(hdb->getName(), "HighDegreeBridge");
}

