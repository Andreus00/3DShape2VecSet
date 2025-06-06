#!/bin/bash
#SBATCH --job-name="vs"
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-9:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32G    # Request more memory
#SBATCH --cpus-per-task=8   # Request more CPUs


/usr/sbin/sshd -D -p 7123 -f /dev/null -h ${HOME}/.ssh/id_ecdsa