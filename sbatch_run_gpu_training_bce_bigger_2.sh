#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J ge-bce-b                  # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=3-00:00             # Allowed runtime in D-HH:MM
#SBATCH --mem=16G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --gres=gpu:4
#SBATCH --output=./logs_ge_bce_bigger_2/myjob-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=./logs_ge_bce_bigger_2/myjob-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=andrea.sanchietti@uni-tuebingen.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls
source ~/.bashrc
conda activate shape2vec
# Compute Phase
# srun env -u SLURM_PROCID python3 main_ae_garmentcode.py --data_path ../GarmentCode/garmentcodedata_v2 --force_occupancy --only_udf # srun will automatically pickup the configuration defined via `#SBATCH` and `sbatch` command line arguments  
srun env -u SLURM_PROCID python3 -m torch.distributed.launch --nproc_per_node=4 \
	--rdzv_endpoint=localhost:29392 main_ae_garmentcode.py \
	--data_path ../GarmentCode/garmentcodedata_v2 \
	--device cuda --batch_size 1 --accum_iter 64 --latent_vec_num 2048 \
	--latent_vec_dim 64 --max_dist 0.1 --warmup_epoch 0 --lr 0.00001 \
	--save_every 1 --epochs 800 --output_dir output_bce_bigger_2 --log_dir output_bce_bigger_2 \
	--point_cloud_size 32768 --num_workers 8 \
	--resume output_bce_bigger/checkpoint-2.pth --start_epoch 3 \
	--grad_weight 0.05

conda deactivate
