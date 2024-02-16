#!/bin/bash
#SBATCH --job-name=pull_request_scrapper        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=cpu         ## ระบุ partition ที่ต้องการใช้งาน

module purge
module load anaconda3
source activate ERAWAN_env

srun python script_flink.py
