### CycleGAN on Simulated Multi-digit MNIST Data
As a proof-of-concept, test if CycleGAN can learn spatial variation from images with multi-digit MNIST


### Procedures
1. prepare two sets of data (domain X and Y) with multi-digit MNIST simulator. They are different in spatial semantics
2. train CycleGAN on the two sets of data. Experiment if CycleGAN can learn to translate between domain X and Y.


### Reference
1. [github] multi-digit MNIST simulator: https://github.com/shaohua0116/MultiDigitMNIST
2. [github] fastai notebook on training CycleGAN: https://github.com/fastai/course-v3/blob/master/nbs/dl2/cyclegan.ipynb

