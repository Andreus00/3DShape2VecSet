#!/bin/bash
#SBATCH --job-name="vs"
#SBATCH --partition=a100-galvani
#SBATCH --time=9:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G    # Request more memory
#SBATCH --cpus-per-task=8   # Request more CPUs


/usr/sbin/sshd -D -p 7123 -f /dev/null -h ${HOME}/.ssh/id_ecdsa