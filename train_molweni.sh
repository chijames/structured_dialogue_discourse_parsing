export CUBLAS_WORKSPACE_CONFIG=:16:8 # to eliminate LSTM non-deterministic
rm -rf *molweni*.pkl # fresh start

if [[ "$1" == "--link_only" ]]
then
    prefix=molweni_link_only_checkpoints
else
    prefix=molweni_checkpoints
fi

GPU_ID=3

for seed in 1 2 3 4 5
do
  for lr in 2e-5
  do
    for epochs in 2
    do
        python main.py --encoder_model ../roberta_base \
          --output_dir "$prefix"/"$seed"_"$lr"_"$epochs"/ \
          --seed $seed \
          --gpu $GPU_ID \
          --data_dir data/molweni \
          --max_num_train_contexts 14 \
          --max_num_dev_contexts 14 \
          --max_num_test_contexts 14 \
          --num_train_epochs $epochs \
          --learning_rate $lr \
          --use_scheduler \
          --fp16 \
          $1
    done
  done
done
