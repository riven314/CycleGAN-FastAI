### CycleGAN on Simulated Multi-digit MNIST Data
As a proof-of-concept, test if CycleGAN can learn spatial variation from images with multi-digit MNIST. I use fastai framework for this project. By the time I am doing this project, I attended lesson 10 of Deep Learning for Coders part 2. This project also serve to practice writing custom callbacks and applying tensorboards in fastai framework. 

### Dataset
MNIST digits with affine transformation. One set of data have one digit per image, another set of data have two digits per image.

## Experiment
I did several experiments. I tried training with flat learning rate as a baseline, and compared its performance against one cycle policy. Though the model trained with one cycle policy have a lower loss. Both models suffer from mode collapse. 

### Reference
1. [github] multi-digit MNIST simulator: https://github.com/shaohua0116/MultiDigitMNIST
2. [github] fastai notebook on training CycleGAN: https://github.com/fastai/course-v3/blob/master/nbs/dl2/cyclegan.ipynb

