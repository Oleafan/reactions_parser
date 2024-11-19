#!/bin/bash
eval "$(conda shell.bash hook)"
conda deactivate 
conda activate molscribe

global_folder="/media/oleg/hard_for_data/papers_for_parse/orglett/orglett_full"
result_folder="/media/oleg/second_ssd/rxn_parsing_data"
hse_main_folder="/home/dchusov/temp_parsing_data"
set -e
for input_idx in 67 72 73 74 91 117 126
do
hse_zip=$hse_main_folder/$input_idx".zip"
echo copy zip file from hse
sshpass -p "UXDk@)a3" scp -P 2222 dchusov@cluster.hpc.hse.ru:$hse_zip $result_folder
echo zip files copied
echo unpacking
unzip $result_folder/$input_idx".zip" -d $result_folder
rm -rf $result_folder/$input_idx".zip"
target=$result_folder/home/dchusov/temp_parsing_data/$input_idx
mv $target $result_folder
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
done
conda deactivate



