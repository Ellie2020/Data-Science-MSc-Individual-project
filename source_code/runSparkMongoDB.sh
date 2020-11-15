#! /bin/bash

# Set working directory, specified as full path or relative path to the directory where the squeue command is executed
#SBATCH --workdir=.

# Set name to display in squeue
#SBATCH --job-name="spark-mongo"

# Send email with task progress
# Use "ALL", "END,FAIL" to skip mail message at start of job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Request resources (keep number of nodes to 1 and tasks to 4-8 to be nice)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

# Separate files for stdout and stderr (where %j gets replaced by job allocation number)
# Use absolute or relative paths, in the latter case paths are relative to the working directory
#SBATCH --output /home/enterprise.internal.city.ac.uk/aczx101/slurm-logs/job-%j-out.txt
#SBATCH --error /home/enterprise.internal.city.ac.uk/aczx101/slurm-logs/job-%j-err.txt

# Test your script with partition set to "test" and few parameters/files
# (job duration limited to 10 minutes), if working move to "normal" partition
#SBATCH --partition=normal

module purge
module load python/intel
module load spark/3.0.0

/usr/bin/time --verbose spark-submit --master local[$SLURM_JOB_CPUS_PER_NODE] --driver-memory 42g --packages org.mongodb.spark:mongo-spark-connector_2.12:2.4.2 --conf "spark.mongodb.input.uri=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
JaccardSIM/DLSH.py
