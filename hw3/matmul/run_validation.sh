#!/bin/bash

./run.sh -v -t 26 831 538 2304 > out/valid/1 &
./run.sh -v -t 9 3305 1864 3494 > out/valid/2 &
./run.sh -v -t 38 618 3102 1695 > out/valid/3 &
./run.sh -v -t 30 1876 3453 3590 > out/valid/4 &
./run.sh -v -t 16 1228 2266 1552 > out/valid/5 &
./run.sh -v -t 2 3347 171 688 > out/valid/6 &
./run.sh -v -t 39 3583 962 765 > out/valid/7 &
./run.sh -v -t 30 2962 373 1957 > out/valid/8 &
./run.sh -v -t 9 3646 2740 3053 > out/valid/9 &
./run.sh -v -t 26 1949 3317 3868 > out/valid/10 &

wait
