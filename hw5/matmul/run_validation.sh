#!/bin/bash

./run.sh -v 831 538 2304 > _out/v/00   &
./run.sh -v 3305 1864 3494 > _out/v/01 &
./run.sh -v 618 3102 1695 > _out/v/02  &
./run.sh -v 1876 3453 3590 > _out/v/03 &
./run.sh -v 1228 2266 1552 > _out/v/04 &
./run.sh -v 347 171 688 > _out/v/05    &
./run.sh -v 3583 962 765 > _out/v/06   &
./run.sh -v 2962 373 1957 > _out/v/07  &
./run.sh -v 3646 2740 3053 > _out/v/08 &
./run.sh -v 1949 3317 3868 > _out/v/09 &

wait
