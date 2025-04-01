#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J garment-encoder                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=32          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=2080-galvani   # Which partition will run your job
#SBATCH --time=3-00:00             # Allowed runtime in D-HH:MM
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --gres=gpu:8
#SBATCH --output=./logs/myjob-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=./logs/myjob-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
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
srun env -u SLURM_PROCID python3 -m torch.distributed.launch main_ae_garmentcode.py --data_path ../GarmentCode/garmentcodedata_v2 --save_every 1 --batch_size 4 --accum_iter 8

conda deactivate
