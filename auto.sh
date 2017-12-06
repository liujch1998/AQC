#!/bin/bash

cpu_mask='0xff'

case $1 in 
	'19260817' )
		cpu_mask='0x1' ;;
	'19980430' )
		cpu_mask='0x2' ;;
	'23333333' )
		cpu_mask='0x4' ;;
	'66666666' )
		cpu_mask='0x8' ;;
	'19980617' )
		cpu_mask='0x10' ;;
	'19701123' )
		cpu_mask='0x20' ;;
	'19700511' )
		cpu_mask='0x40' ;;
	'20091001' )
		cpu_mask='0x80' ;;
esac

for i in {10..24..2}; do
	taskset $cpu_mask python batch.8.py max_clique $i $1
done

