#!/bin/bash

mkdir lib

# googletest framework

wget https://github.com/amusaafir/googletest/archive/master.zip -P lib

unzip lib/master.zip -d lib

rm lib/master.zip