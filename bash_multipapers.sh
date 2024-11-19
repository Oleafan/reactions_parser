#!/bin/bash
eval "$(conda shell.bash hook)"
conda deactivate 
conda activate molscribe
global_folder="/media/oleg/second_ssd/pdf_for_parse/obc_papers"
temp_folder="/media/oleg/second_ssd/temp_files_parser"
# echo creating temp folders and making pngs and texts
# for subfolder in $(ls $global_folder) 
# do 
# folder_path=$global_folder/$subfolder 
# python /media/oleg/second_ssd/reaction_parser/create_text_image.py $folder_path $temp_folder
# echo $folder_path
# done
# echo "running opsin"
# for subfolder in $(ls $temp_folder) 
# do
# candidates_file=$temp_folder/$subfolder/candidates.txt
# smiles_file=$temp_folder/$subfolder/smiles.txt
# echo $subfolder
# java -jar /media/oleg/second_ssd/reaction_parser/opsin280.jar -s -osmi $candidates_file $smiles_file > /dev/null 2>&1
# done
# echo "Recognising molecules"
# python /media/oleg/second_ssd/ocsr_molscribe_mp/run_molscribe_ocsr.py $temp_folder 2>&1 > /dev/null 
# echo "processing text"
# python process_text_multifolder.py $temp_folder #> /dev/null 2>&1 
echo "protocols classification"
python classify_protocols_multibatch.py $temp_folder #> /dev/null 2>&1 
conda deactivate
conda activate mistral
echo "parse protocols"
python /media/oleg/second_ssd/rubaha_nlp_mistral-main/eval_mistral_multibatch_multifolder.py $temp_folder
conda deactivate
conda activate base
echo "create reactions"
python /media/oleg/second_ssd/reaction_parser/reactions_processing.py $temp_folder #> /dev/null 2>&1 
conda deactivate