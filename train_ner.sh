export MAX_LENGTH=128
export BERT_MODEL="bert-base-multilingual-cased"
# date=$(date +'%d_%m_%Y')
today=$(date +'%Y_%b_%d')
brat_folder="${1:-'/home/compute-gdi/brat/data/relations/Экономика/*/*.ann'}"
#echo $brat_folder
cp -r /home/compute-gdi/brat/data/trade_news_2021 /home/compute-gdi/brat/data/relations/Экономика/
datadir="${2:-${today}_ner}"
output_dir="${3:-$HOME/nlp_dl_server/${today}_ner_model/}"
#python3 get_annotated_files_from_brat.py $brat_folder $datadir 
# remove tabs
grep -v "^#" $datadir/ner_data/train.txt | cut -f 1,2 | tr '\t' ' ' > train.txt.tmp
grep -v "^#" $datadir/ner_data/valid.txt   | cut -f 1,2 | tr '\t' ' ' >   dev.txt.tmp
grep -v "^#" $datadir/ner_data/test.txt  | cut -f 1,2 | tr '\t' ' ' >  test.txt.tmp
# preprocess
python3 ner/preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
python3 ner/preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
python3 ner/preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
# generate labels
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
#cat train.txt dev.txt test.txt | grep -v "^#" ner_data/train.txt | cut -f 2,3 | tr '\t' ' ' | sort | uniq > labels.txt
cat labels.txt
#
export OUTPUT_DIR=$output_dir
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1
# train
python3 ner/run_ner.py \
--data_dir ./ \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir
--fp16
#--model_type bert \

