%%bash
export CUDA_VISIBLE_DEVICES=0
model_root=checkpoints
data_path=/media/ellen/datasets
output_dir=out

# vtab-structured: dmlab
# base_lr = 1.0
# lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
for seed in "42"; do
    python train_weakly.py \
        --config-file configs/prompt/pascal.yaml \
        MODEL.TYPE "ssl-vit-seg" \
        DATA.BATCH_SIZE "64" \
        MODEL.TRANSFER_TYPE "prompt" \
        MODEL.PROMPT.NUM_TOKENS "50" \
        MODEL.PROMPT.DEEP "False" \
        MODEL.PROMPT.DROPOUT "0.0" \
        DATA.FEATURE "mae_vitb16" \
        DATA.NUMBER_CLASSES "21" \
        DATA.NO_TEST "True" \
        SOLVER.BASE_LR "0.25" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}" \
        OUTPUT_DIR "${output_dir}/seed${seed}" \
        BARLOW.LAMBD "0.005"
done