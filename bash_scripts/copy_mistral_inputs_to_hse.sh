#!/bin/bash
global_folder="/media/oleg/second_ssd/rxn_parsing_data"

for parse_id in 67 72 73 74 91 100 101 117 126
do
    #copy all folders like parse_id_XXX to temp_folder with name parse_id
    work_folder=$global_folder/$parse_id
    temp_folder=$work_folder/$parse_id
    if [ ! -d "$temp_folder" ]; then  mkdir $temp_folder; fi
    
    for sub_id in 1 2 3 4 5 6 7 8 9 10
    do
        subwork_folder=$work_folder/"$parse_id"_"$sub_id"
        if [ -d "$subwork_folder" ]; then  cp -r $subwork_folder $temp_folder; fi
    done
    
    #copy to hse server
    hse_folder="/home/dchusov/mistral_temp"
    sshpass -p "UXDk@)a3" scp -P 2222 -r $temp_folder dchusov@cluster.hpc.hse.ru:$hse_folder
    
    #copy sbatch files to work mistral dir
    mistral_dir="/home/dchusov/mistral/rubaha_nlp_mistral-main"
    for sub_id in 1 2 3 4 5 6 7 8 9 10
    do 
        subwork_folder=$work_folder/"$parse_id"_"$sub_id"
        if [ -d "$subwork_folder" ]; then  
            sbatch_file=$subwork_folder/"$parse_id"_"$sub_id".sbatch
            sshpass -p "UXDk@)a3" scp -P 2222 $sbatch_file dchusov@cluster.hpc.hse.ru:$mistral_dir
        fi
    done
done
