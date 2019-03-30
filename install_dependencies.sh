#!/bin/bash

mkdir lib

# Snap-stanford

wget https://github.com/amusaafir/snap/archive/master.zip -P lib

unzip lib/master.zip -d lib

rm lib/master.zip

mv lib/snap-master lib/snap

(cd lib/snap && make all)

# googletest framework

wget https://github.com/amusaafir/googletest/archive/master.zip -P lib

unzip lib/master.zip -d lib

rm lib/master.zip

mv lib/googletest-master lib/googletest