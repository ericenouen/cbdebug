model=$1
######################################

if [ "$model" == "pipnet" ]; then
    cd pipnet
    python -m debug.evaluate --dataset Waterbirds --log_dir ./runs/eval --state_dict_dir_net ./runs
    python -m debug.evaluate --dataset MetaShift --log_dir ./runs/eval --state_dict_dir_net ./runs

######################################

elif [ "$model" == "pcbm" ]; then
    cd pcbm
    datasets=("Waterbirds" "MetaShift" "CelebA")
    for dataset in "${datasets[@]}"; do
        python -m debug.evaluate --dataset ${dataset} --backbone-dir concept_banks/ --backbone-name clip:ViT-L-14
    done
fi