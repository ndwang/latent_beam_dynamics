#!/bin/bash
#SBATCH --job-name=lbd_compare_ar
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# AR INFERENCE COMPARISON — three architectures
#   Tracking d512  (lr=3e-3, scan_arch_tracking_d_model512_lr3e-3_260418_1324)
#   DualStream d256 (scan_arch_dualstream_d_model256_260418_0738)
#   Lattice d128 L12 (scan_direct_depth_d128_L12_260414_1314)
# All evaluated on encoded_sectioned_10k val split.
# ============================================

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate lbd

python scripts/compare_ar.py \
    runs/scan_arch_tracking_d_model512_lr3e-3_260418_1324/scan_arch_tracking_d_model512_lr3e-3_260418_1324_best.pth \
    runs/scan_arch_dualstream_d_model256_260418_0738/scan_arch_dualstream_d_model256_260418_0738_best.pth \
    runs/scan_direct_depth_d128_L12_260414_1314/scan_direct_depth_d128_L12_260414_1314_best.pth \
    --data data/encoded_sectioned_10k \
    --output eval_compare_ar \
    --include-tf
