
#! /bin/bash

for seed in 0; do
  for experiment in 2; do
    for residual in 0; do
      for n_layers in 3; do
        for n_neurons in 64; do
          for N_trj in 1000; do
            for T_trj in 100; do
              for N_dim in 4 10; do
                for latent_size in 4 16 64 512; do
                  for reg_energy in 0.5; do
                    sbatch --export=model=0,experiment=$experiment,N_trj=$N_trj,T_trj=$T_trj,latent_size=$latent_size,N_dim=$N_dim,n_neurons=$n_neurons,n_layers=$n_layers,residual=$residual,seed=$seed,reg_energy=$reg_energy hyperparams_search.sbatch
                  done
                  sbatch --export=model=1,experiment=$experiment,N_trj=$N_trj,T_trj=$T_trj,latent_size=$latent_size,N_dim=$N_dim,n_neurons=$n_neurons,n_layers=$n_layers,residual=$residual,seed=$seed,reg_energy=0.5 hyperparams_search.sbatch
                done
              done
            done
          done
        done
      done
    done
  done
done













