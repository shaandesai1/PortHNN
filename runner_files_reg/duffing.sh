#!/bin/bash

#mass spring
python main_reg.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 10.01 -dt 0.01 -noise_std 0.1 -dname duffing -type 1 -noise_std 0 -batch_size 200 -learning_rate 1e-3
