python create_brat_tacred_dataset.py --source=$1 --dest_dir=$2;
python ~/SpanBERT/code/run_tacred.py \
	--do_train   \
	--do_eval   \
	--data_dir $2    \
	--model "bert-base-multilingual-cased" \
	--train_batch_size 32   \
	--eval_batch_size 32   \
	--learning_rate 2e-5   \
	--num_train_epochs 10   \
	--max_seq_length 128   \
	--output_dir $2


