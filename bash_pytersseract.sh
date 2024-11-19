#!/bin/bash
eval "$(conda shell.bash hook)"
conda deactivate 
conda activate molscribe

global_folder="/media/oleg/hard_for_data/papers_for_parse/orglett/orglett_full"
result_folder="/media/oleg/second_ssd/rxn_parsing_data"
hse_main_folder="/home/dchusov/temp_parsing_data"

file='/media/oleg/second_ssd/rxn_parsing_data/file_index.txt'

# hse_folder=$hse_main_folder/36
# sshpass -p "UXDk@)a3" scp -P 2222 -r dchusov@cluster.hpc.hse.ru:$hse_folder $result_folder &

for input_idx in 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100     
do
wait
temp_folder=$result_folder/$input_idx
echo running pytesseract
python /media/oleg/second_ssd/ocsr_molscribe_mp/finalize_ocr_pytesseract.py $temp_folder
echo "running opsin"
python /media/oleg/second_ssd/reaction_parser/running_opsing.py $temp_folder
echo "delete temp pngs"
python remove_pngs.py $temp_folder
echo download next folder
let next_idx=$input_idx+1
hse_folder=$hse_main_folder/$next_idx
sshpass -p "UXDk@)a3" scp -P 2222 -r dchusov@cluster.hpc.hse.ru:$hse_folder $result_folder &
echo "processing text"
python process_text_multifolder.py $temp_folder #> /dev/null 2>&1 
echo "protocols classification"
python classify_protocols_multibatch.py $temp_folder #> /dev/null 2>&1 
echo "group protocols"
python join_protocols.py $temp_folder
done
wait
input_idx=101
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
conda deactivate