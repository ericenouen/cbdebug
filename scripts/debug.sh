model=$1
######################################

if [ "$model" == "pipnet" ]; then
    cd pipnet

    datasets=("Waterbirds" "MetaShift")
    for dataset in "${datasets[@]}"; do
        for seed in {1..3}; do
            python main.py --dataset ${dataset} --seed $seed --epochs_pretrain 10 --batch_size 64 --freeze_epochs 10 --epochs 60 --log_dir ./runs/pipnet_${dataset,,}${seed}
        done
    done

######################################

elif [ "$model" == "pcbm" ]; then
    cd pcbm
    
    datasets=("Waterbirds" "MetaShift" "CelebA")
    for dataset in "${datasets[@]}"; do
        python get_predecidedconcepts_multimodal.py --classes ${dataset} --backbone-name clip:ViT-L-14 --out-dir concept_banks/
        for seed in {1..3}; do
            python train_pcbm.py --concept-bank concept_banks/multimodal_concept_clip:ViT-L-14_${dataset}.pkl --seed $seed \
            --dataset ${dataset} --backbone-dir concept_banks/ --backbone-name clip:ViT-L-14 --out-dir concept_banks/original_models --lam=2e-2
        done
    done
fi

######################################