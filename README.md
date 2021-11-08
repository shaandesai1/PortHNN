# PortHNN
Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent Dynamical Systems

We show that an embedded Port-HNN in neural networks is significantly more performant than existing approaches at learning from explicit, non-autonomous time-dependent physical systems.

To run this code, all you need is torch.

cd into the main directory, then type in:

```
./runner_files_reg/run_all_methods.sh 
```

It will run all methods.

A separate file is designated for the coupled system and runs in its own jupyter notebook.


Note: The default configuration will train all the methods with an embedded integrator (RK4) for a single-step integration i.e. t to t+1. To change the training regime to use gradients, edit:

```
parser.add_argument('-embed_integ','--embed_integ',action="store_false")
```

to

```
parser.add_argument('-embed_integ','--embed_integ',action="store_true")
```

.

It is possible to extend the method to multi-step integration via neuralODE [needs implementation]


