#!/bin/bash

#mass spring
python main_reg.py -ni 20000 -n_test_traj 1 -n_train_traj 20 -dname duffing -type 3 -noise_std 0 -batch_size 100 -learning_rate 1e-3
