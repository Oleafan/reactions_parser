#!/bin/bash

hse_main_folder="/home/dchusov/mistral_temp"
local_folder="/media/oleg/second_ssd/rxn_parsing_data/processed"
for parse_id in 34 46 67 72 73 74 91 117 126
do 
    parse_folder=$hse_main_folder/$parse_id
    sshpass -p "UXDk@)a3" scp -P 2222 -r dchusov@cluster.hpc.hse.ru:$parse_folder $local_folder
done



