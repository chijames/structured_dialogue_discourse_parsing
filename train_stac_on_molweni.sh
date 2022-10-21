export CUBLAS_WORKSPACE_CONFIG=:16:8 # to eliminate LSTM non-deterministic
rm -rf *stac*.pkl # fresh start

if [[ "$1" == "--link_only" ]]
then
    prefix=stac_on_molweni_link_only_checkpoints
else
    prefix=stac_on_molweni_checkpoints
fi

GPU_ID=2

for seed in 1 2 3 4 5
do
  for lr in 2e-5
  do
    for epochs in 6
    do
        python main.py --encoder_model ../roberta_base \
          --output_dir "$prefix"/"$seed"_"$lr"_"$epochs"/ \
          --seed $seed \
          --gpu $GPU_ID \
          --data_dir data/stac \
          --test_data_dir data/molweni \
          --max_num_train_contexts 20 \
          --max_num_dev_contexts 14 \
          --max_num_test_contexts 14 \
          --num_train_epochs $epochs \
          --train_batch_size 2 \
          --learning_rate $lr \
          --use_scheduler \
          --fp16 \
          $1
    done
  done
done
