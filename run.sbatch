#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=habitat_example
#SBATCH --mail-type=END
#SBATCH --mail-user=pr2257@nyu.edu

module purge

singularity exec --nv \
			--bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
	    --overlay /scratch/pr2257/images/overlay-50G-10M.ext3:ro \
   /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate habitat; python /scratch/pr2257/deep-learning/project-2/habitat-sim/examples/example.py --scene /scratch/pr2257/deep-learning/project-2/habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
