#!/bin/bash
eval "$(conda shell.bash hook)"
conda deactivate 
conda activate molscribe

result_folder="/media/oleg/second_ssd/rxn_parsing_data_joc"
hse_main_folder="/home/dchusov/temp_parsing_data"
hse_input_folder="/home/dchusov/inp_data/joc"

file=$result_folder/"file_index.txt"

set -e

for input_idx in $(cat $file)
    do 
        #we should run ocr tasks but not more than 6 more then current inp_idx
        #0_1, 2_3, 4_5 already started
        let next_start_idx=$input_idx+5
        let extra_start_idx=$next_start_idx+1
        next_ocr_sbatch=$next_start_idx"_"$extra_start_idx"_ocr.sbatch"
        if sshpass -p "UXDk@)a3" ssh dchusov@cluster.hpc.hse.ru -p2222 "test -e /home/dchusov/temp_parsing_data/sbatch_files/$next_ocr_sbatch"
        then 
            sshpass -p "UXDk@)a3" ssh dchusov@cluster.hpc.hse.ru -p2222 "cd /home/dchusov/temp_parsing_data/sbatch_files; sbatch $next_ocr_sbatch"
        fi
        
        hse_folder=$hse_main_folder/$input_idx
        
        #try download next folder
        let next_idx=$input_idx+1
        hse_next_folder=$hse_main_folder/$next_idx    
        
        finish_next_file=$hse_next_folder/"finished.log"
        
        if sshpass -p "UXDk@)a3" ssh dchusov@cluster.hpc.hse.ru -p2222 "test -e $finish_next_file"
        then 
            hse_next_zip=$hse_main_folder/"sbatch_files"/$next_idx".zip" #check is it a correct path
            echo "copy next zip file from hse while working with this file"
            sshpass -p "UXDk@)a3" scp -P 2222 dchusov@cluster.hpc.hse.ru:$hse_next_zip $result_folder &
        fi 
       
        echo unpacking
        unzip $result_folder/$input_idx".zip" -d $result_folder
        
        target=$result_folder/home/dchusov/temp_parsing_data/$input_idx
        mv $target $result_folder
        
        #remove zip from this computer
        rm -rf $result_folder/$input_idx".zip"
        
        temp_folder=$result_folder/$input_idx
        echo running pytesseract
        python /media/oleg/second_ssd/ocsr_molscribe_mp/finalize_ocr_pytesseract.py $temp_folder
        echo "running opsin"
        python /media/oleg/second_ssd/reaction_parser/running_opsing.py $temp_folder
        echo "delete temp pngs"
        python remove_pngs.py $temp_folder
        echo "processing text"
        python process_text_multifolder.py $temp_folder #> /dev/null 2>&1 
        echo "protocols classification"
        python classify_protocols_multibatch.py $temp_folder #> /dev/null 2>&1 
        echo "group protocols"
        python join_protocols.py $temp_folder
        

        #removal of these folders from hse:
        hse_zip=$hse_main_folder/"sbatch_files"/$input_idx".zip" #check is it a correct path
        sshpass -p "UXDk@)a3" ssh dchusov@cluster.hpc.hse.ru -p2222 "rm -rf $hse_zip"
        sshpass -p "UXDk@)a3" ssh dchusov@cluster.hpc.hse.ru -p2222 "rm -rf $hse_folder"
        
        #copy mistral inputs
        #copy all folders like parse_id_XXX to temp_folder with name parse_id and run them
        work_folder=$result_folder/$input_idx
        temp_folder=$work_folder/$input_idx
        if [ ! -d "$temp_folder" ]; then  mkdir $temp_folder; fi

        for sub_id in 1 2 3 4 5 6 7 8 9 10
        do
            subwork_folder=$work_folder/"$input_idx"_"$sub_id"
            if [ -d "$subwork_folder" ]; then  cp -r $subwork_folder $temp_folder; fi
        done

        #copy to hse server
        hse_folder_mistral="/home/dchusov/mistral_temp"
        sshpass -p "UXDk@)a3" scp -P 2222 -r $temp_folder dchusov@cluster.hpc.hse.ru:$hse_folder_mistral

        #copy sbatch files to work mistral dir
        mistral_dir="/home/dchusov/mistral/rubaha_nlp_mistral-main"
        for sub_id in 1 2 3 4 5 6 7 8 9 10
        do 
            subwork_folder=$work_folder/"$input_idx"_"$sub_id"
            if [ -d "$subwork_folder" ]; then  
                sbatch_file=$subwork_folder/"$input_idx"_"$sub_id".sbatch
                sshpass -p "UXDk@)a3" scp -P 2222 $sbatch_file dchusov@cluster.hpc.hse.ru:$mistral_dir
                #run mistral sbatch
                sshpass -p "UXDk@)a3" ssh dchusov@cluster.hpc.hse.ru -p2222 "cd $mistral_dir; sbatch "$input_idx"_"$sub_id".sbatch"
            fi
        done
        
        echo "waiting for finishing of the download if it was started"
        wait
        if [ -f $result_folder/$next_idx".zip" ]
            then echo "next idx already downloaded"
        else
            while :
            do
                if sshpass -p "UXDk@)a3" ssh dchusov@cluster.hpc.hse.ru -p2222 "test -e $finish_next_file"; then break; fi 
                echo "waiting for parsing of pngs"
                sleep 1800 #check this every 30 minutes
            done     
            hse_next_zip=$hse_main_folder/"sbatch_files"/$next_idx".zip" #check is it a correct path
            echo "copy next zip file from hse"
            sshpass -p "UXDk@)a3" scp -P 2222 dchusov@cluster.hpc.hse.ru:$hse_next_zip $result_folder 
            echo "zip files copied"
        fi
        
    done
