#!/bin/bash

#mass spring
python main_reg.py -ni 20000 -n_test_traj 25 -n_train_traj 5 -tmax 20.01 -dt 0.01 -dname relativity -type 1 -noise_std 0 -batch_size 200 -learning_rate 1e-3
