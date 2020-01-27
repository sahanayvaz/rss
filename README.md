#### RSS
Random Sparse Skipped (RSS) layer aims to reduce the number of parameters between the convolutional and fully connected layers which are still widely used in deep reinforcement learning benchmark such as Nature-CNN.

RSS introduces a randomly created sparse mask with skipped sparse connections between layers. Using this architecture, we can reduce the number of parameters in the fully connected layers by more than 96% without suffering from a significant performance loss.

We experimented with using the remaining connections to pack more tasks such as learning to play Mario Level1-1, Level2-1, Level3-1 and etc. RSS could be used to pack multiple reinforcement learning tasks into a single architecture by utilizing predefined sparse connections for each task.

All the experiments in this repository uses Proximal Policy Optimization with clipped variant.