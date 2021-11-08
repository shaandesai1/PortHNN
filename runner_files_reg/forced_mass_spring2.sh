#!/bin/bash

#mass spring
python main_reg.py -ni 10000 -n_test_traj 25 -n_train_traj 20 -tmax 10.01 -dt 0.01 -dname forced_mass_spring -noise_std 0 -type 2 -batch_size 200 -learning_rate 1e-3
