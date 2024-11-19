#!/bin/bash

file='/media/oleg/second_ssd/rxn_parsing_data/34_99_inputs.txt'
for folder in $(cat $file)
do
sshpass -p "UXDk@)a3" scp -P 2222 -r $folder dchusov@cluster.hpc.hse.ru:/home/dchusov/inp_data/orglett
done