#!/bin/bash
eval "$(conda shell.bash hook)"
conda deactivate 
conda activate molscribe
global_folder="/media/oleg/second_ssd/pdf_for_parse/joc_papers_si_400"
temp_folder="/media/oleg/second_ssd/rxn_parsing_081024"
#log file
# echo "Start processing " 
# echo "Start time: " $(date +%s)
# echo "creating temp folders and making pngs and texts" 
# python /media/oleg/second_ssd/reaction_parser/create_text_image_mp.py $global_folder $temp_folder &
# sleep 90
# echo "running opsin in parallel with creation of text and images"
# python running_opsing.py $temp_folder &
# sleep 30
# echo "Recognising molecules in parallel with image creation"
# python /media/oleg/second_ssd/ocsr_molscribe_mp/run_molscribe_ocsr_parallel.py $temp_folder 2>&1 > /dev/null 
# wait
echo "processing text"
python process_text_multifolder.py $temp_folder #> /dev/null 2>&1 
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
echo "End time: " $(date +%s)