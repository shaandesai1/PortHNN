#!/bin/bash

#mass spring
python main_reg.py -ni 20000 -n_test_traj 3 -n_train_traj 1 -tmax 10001 -dt 1 -dname relativity -type 2 -noise_std 0 -batch_size 200 -learning_rate 1e-3
