#!/bin/bash
#
#SBATCH --job-name=MetMatNN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=myPartition
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=myEmail@email.com

module purge
module load python
module load gcc

cd /myDir
pip install -r requirements.txt
python forwardNN.py --output_path /myDir/results

