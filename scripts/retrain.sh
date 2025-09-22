model=$1
######################################

if [ "$model" == "pipnet" ]; then
    cd pipnet
    algorithms=("Remove" "Retrain" "Augment" "ProtoPDebug" "PermutationWeighting" "CBDebug")
    runs=("run_1" "run_2" "run_3" "run_4" "run_5" "run_6")
    for run in "${runs[@]}"; do
        num=${run##*_}
        seed=$(( (num - 1) % 3 + 1 ))
        for algorithm in "${algorithms[@]}"; do
            python -m debug.finetune \
                --dataset Waterbirds \
                --seed ${seed} --algorithm "$algorithm" \
                --batch_size 64 \
                --freeze_epochs 0 --epochs 30 \
                --log_dir ./runs/${run}/pipnet_waterbirds_finetune_$algorithm$i \
                --state_dict_dir_net ./runs/pipnet_waterbirds${seed}/checkpoints/net_trained_last \
                --prune_weight_path debug/user_study/${run}/pip_pruneweight_waterbirds_${seed}.pth \
                --lr_net 0.00001 --lr_block 0.00001 --lr 0.05
            python -m debug.finetune \
                --dataset MetaShift \
                --seed ${seed} --algorithm "$algorithm" \
                --batch_size 64 \
                --freeze_epochs 0 --epochs 30 \
                --log_dir ./runs/${run}/pipnet_metashift_finetune_$algorithm$i \
                --state_dict_dir_net ./runs/pipnet_metashift${seed}/checkpoints/net_trained_last \
                --prune_weight_path debug/user_study/${run}/pip_pruneweight_metashift_${seed}.pth \
                --lr_net 0.00001 --lr_block 0.00001 --lr 0.05
        done
    done

######################################

elif [ "$model" == "pcbm" ]; then
    cd pcbm
    algorithms=("Remove" "Retrain" "Augment" "PermutationWeighting" "CBDebug")
    runs=("run_1" "run_2" "run_3" "run_4" "run_5" "run_6")
    # runs=("llm_run_1" "llm_run_2" "llm_run_3")
    for run in "${runs[@]}"; do
        num=${run##*_}
        seed=$(( (num - 1) % 3 + 1 ))
        dataset="Waterbirds"
        for algorithm in "${algorithms[@]}"; do
            python -m debug.finetune --algorithm "$algorithm" --dataset ${dataset} \
            --seed $seed --lam=2e-2 --out-dir concept_banks/${run}/ --backbone-dir concept_banks/ --backbone-name clip:ViT-L-14 \
            --pcbm-path concept_banks/original_models/pcbm_${dataset}__clip:ViT-L-14__multimodal_concept_clip:ViT-L-14_${dataset}__lam:0.02__alpha:0.99__seed:${seed}.ckpt \
            --prune_weight_path debug/user_study/${run}/pcbm_pruneweight_waterbirds_${seed}.pth \
            --concept-bank concept_banks/multimodal_concept_clip:ViT-L-14_${dataset}.pkl
        done
        dataset="MetaShift"
        for algorithm in "${algorithms[@]}"; do
            python -m debug.finetune --algorithm "$algorithm" --dataset ${dataset} \
            --seed $seed --lam=2e-2 --out-dir concept_banks/${run}/ --backbone-dir concept_banks/ --backbone-name clip:ViT-L-14 \
            --pcbm-path concept_banks/original_models/pcbm_${dataset}__clip:ViT-L-14__multimodal_concept_clip:ViT-L-14_${dataset}__lam:0.02__alpha:0.99__seed:${seed}.ckpt \
            --prune_weight_path debug/user_study/${run}/pcbm_pruneweight_metashift_${seed}.pth \
            --concept-bank concept_banks/multimodal_concept_clip:ViT-L-14_${dataset}.pkl
        done
        dataset="CelebA"
        for algorithm in "${algorithms[@]}"; do
            python -m debug.finetune --algorithm "$algorithm" --dataset ${dataset} \
            --seed $seed --lam=2e-2 --out-dir concept_banks/${run}/ --backbone-dir concept_banks/ --backbone-name clip:ViT-L-14 \
            --pcbm-path concept_banks/original_models/pcbm_${dataset}__clip:ViT-L-14__multimodal_concept_clip:ViT-L-14_${dataset}__lam:0.02__alpha:0.99__seed:${seed}.ckpt \
            --prune_weight_path debug/user_study/${run}/pcbm_pruneweight_celeba_${seed}.pth \
            --concept-bank concept_banks/multimodal_concept_clip:ViT-L-14_${dataset}.pkl
        done
    done
fi
######################################