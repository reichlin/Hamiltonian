#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --constrain='arwen|balrog|eowyn|galadriel|gondor|khazadum|rivendell|smaug'

# arwen|balrog|belegost|eowyn|galadriel|gondor|khazadum|rivendell|shelob|shire|smaug


source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate hessian_torch

# export PYTHONPATH=$PYTHONPATH:/Midgard/home/areichlin/MetricRL

python3 /Midgard/home/areichlin/Hamiltonian/main.py \
--model ${model} \
 --experiment ${experiment} \
  --N_trj ${N_trj} \
   --T_trj ${T_trj} \
    --latent_size ${latent_size} \
     --N_dim ${N_dim} \
      --n_neurons ${n_neurons} \
       --n_layers ${n_layers} \
        --residual ${residual} \
         --reg_energy ${reg_energy} \
          --seed ${seed}
