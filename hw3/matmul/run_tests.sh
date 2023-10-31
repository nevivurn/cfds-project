#!/bin/bash

run() {
	./run.sh -v -t $1 -n 10 4096 4096 4096 | tee out/guided/out_$1
}

run 1 &
run 2 &
run 4 &
run 8 &
run 12 &
run 16 &
run 16 &
run 20 &
run 24 &
run 28 &
run 32 &
run 40 &
run 48 &
run 56 &
run 64 &
run 80 &
run 96 &
run 112 &
run 128 &
run 160 &
run 192 &
run 224 &
run 256 &

wait
