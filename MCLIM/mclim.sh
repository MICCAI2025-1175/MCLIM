EXP_NAME="MCLIM"
EXP_DIR="MCLIM/pretrain/output_${EXP_NAME}"

conda init bash
source ~/.bashrc
conda activate your-env

cd your-path


echo "============== Pretraining starts =============="
touch ~/wait1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=4,5,6,7 python launch.py \
  --main_py_relpath main_v3.py \
  --exp_name "${EXP_NAME}" \
  --exp_dir "${EXP_DIR}" \
  --num_nodes=1 \
  --ngpu_per_node=4 \
  --node_rank=0 \
  --master_address=128.0.1.3 \
  --master_port=5200 \
  --data_path=MCLIM/adni_mni152_affine_clean_v2.txt \
  --opt=adamw \
  --bs=12 \
  --ep=200 \
  --wp_ep=10 \
  --input_size=96 \
  --dataloader_workers=10 \
  --base_lr=1e-4 \
  --wd=0.2 \
  --mim_ratio=0.75 \
  --patch_size=16 \
  --weight_recon=1.0 \
  --resume_from=MCLIM/pretrain/MCLIM/unet_still_pretraining.pth
echo "============== Pretraining ends =============="
rm ~/wait1