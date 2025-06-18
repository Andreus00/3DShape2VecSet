#!/bin/bash
#SBATCH --job-name="vs"
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-01:00
#SBATCH --gres=gpu:2
#SBATCH --mem=16G    # Request more memory
#SBATCH --cpus-per-task=8   # Request more CPUs
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=andrea.sanchietti@uni-tuebingen.de   # Email to which notifications will be sent
#SBATCH --output=./vslogs/myjob-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=./vslogs/myjob-%j.err        # File to which STDERR will be written - make sure this is not on $HOME


/usr/sbin/sshd -D -p 7123 -f /dev/null -h ${HOME}/.ssh/id_ecdsa
