# The Path to Multi GPU Systems

Markov Decision Processes (MDPs) are a powerful decision making framework and modern applications can be extremely large with respect to the size of the state and action spaces. Graphical Processing Units (GPUs) can improve algorithms efficiency by introducing parallel computation capabilities. Furthermore, there has been significant recent progress in developing methods for systems with multiple GPUs which enhance the memory size and computational performance. We develop a novel method and investigate the performance of using multi-GPU systems to solve MDPs through Value Iteration. We compare the performance to both single GPU systems and CPU-only systems.

## Running experiments

In order to run an experiment we must first compile the different GPU CUDA files: 

```
nvcc -o multi multi.cu
nvcc -o single single.cu
```

Then we can run an experiment with the specified action space size, state space size and iteration number:

```
./multi -A 1024 -S 1024 -i 100 
./single -A 1024 -S 1024 -i 100 
```

## Results

Below we include some of the results of the experiments:

![Alt text](IterationsExp.png?raw=true "Iterations")

![Alt text](StateExp.png?raw=true "State")

![Alt text](ActionExp.png?raw=true "Action")

