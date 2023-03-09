#!/bin/bash

x=8
y=30

step=2

filepath="example/taskset_08022022/"
# declare -a program=("sbeet_mg" "lcf" "bcf")
declare -a program=("ld" "lcf" "bcf" "mg-jm")
# declare -a program=("mg-offline")

for val in "${program[@]}"; do
	mkdir output/taskset_08022022/${val}
	for i in `seq ${x} $step ${y}`; do
      	# su=$(printf '%02d' "${i}")
      	python3 main.py -u ${i} -d 15 -p ${val}
		# python3 main.py -u ${i} -d 300 -p ${val} -m 1
		sleep 1
		killall python3
	done
	sleep 0.2
	killall python3
	echo "${val} ${i} finished"
done