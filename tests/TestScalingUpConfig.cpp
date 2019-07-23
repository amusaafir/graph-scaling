//
// Created by Ahmed on 3-4-19.
//

#include <gtest/gtest.h>
#include "../scaling/scale-up/ScalingUpConfig.h"

TEST(ScalingUpConfig, testRemainder3_05) {
    ScalingUpConfig conf(3, 0.5, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder3_5__05) {
    ScalingUpConfig conf(3.5, 0.5, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder4_7__05) {
    ScalingUpConfig conf(4.7, 0.5, NULL, NULL);

    ASSERT_TRUE(conf.hasSamplingRemainder());

    ASSERT_EQ(conf.getRemainder(), 0.2f);
}

TEST(ScalingUpConfig, testRemainder2_1__01) {
    ScalingUpConfig conf(2.1, 0.1, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder2_0__02) {
    ScalingUpConfig conf(2.0, 0.2, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder3_0__03) {
    ScalingUpConfig conf(3.0, 0.3, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder7_0__07) {
    ScalingUpConfig conf(7.0, 0.7, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder7_5__05) {
    ScalingUpConfig conf(7.5, 0.5, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder232__05) {
    ScalingUpConfig conf(232, 0.5, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder5__025) {
    ScalingUpConfig conf(5, 0.25, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder20__08) {
    ScalingUpConfig conf(20, 0.8, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder10__04) {
    ScalingUpConfig conf(10, 0.4, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder5__05) {
    ScalingUpConfig conf(5, 0.5, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder10__05) {
    ScalingUpConfig conf(10, 0.5, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}

TEST(ScalingUpConfig, testRemainder36__06) {
    ScalingUpConfig conf(36, 0.6, NULL, NULL);

    ASSERT_FALSE(conf.hasSamplingRemainder());
}