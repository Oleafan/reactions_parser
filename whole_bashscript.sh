#!/bin/bash
echo "Create text files and page images"
eval "$(conda shell.bash hook)"
conda deactivate 
conda activate molscribe
folder_path="/media/oleg/second_ssd/pdf_for_parse/gen_2"
doi="gen/2"
work_folder=/media/oleg/second_ssd/temp_files_parser/${doi//\//_}
python /media/oleg/second_ssd/reaction_parser/create_text_image.py $doi $folder_path
echo "running opsin"
candidates_file=$work_folder/candidates.txt
smiles_file=$work_folder/smiles.txt
java -jar /media/oleg/second_ssd/reaction_parser/opsin280.jar -s -osmi $candidates_file $smiles_file > /dev/null 2>&1
echo "Recognising molecules"
python /media/oleg/second_ssd/OdanReactOCSR/run_code.py $work_folder 2>&1 > /dev/null 
echo "processing text"
python process_text.py $work_folder  > /dev/null 2>&1
echo "protocols classification"
python classify_protocols.py $work_folder
conda deactivate
conda activate mistral
echo "parse protocols"
python /media/oleg/second_ssd/rubaha_nlp_mistral-main/eval_mistral_multibatch.py $work_folder
conda deactivate
