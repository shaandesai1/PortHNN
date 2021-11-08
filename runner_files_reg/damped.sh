#!/bin/bash

#mass spring
python main_reg.py -ni 5000 -n_test_traj 25 -n_train_traj 25 -tmax 30.1 -dt 0.1 -dname damped -noise_std 0 -batch_size 200 -learning_rate 1e-3
