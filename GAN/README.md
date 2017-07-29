# Generative Adversarial Networks

Discovered by Ian J. Goodefellow, and presented to the world in 2014, they are one of the hottest things in Machine Learning right now. They are deep convolution or sequence based models which learn using the simplest of ideas, backed up by a good amount of theory ofcourse. 

The idea is that in order to model a distribution we try to find a function approximators (neural networks) which learn to map random samples to a better distrubition. The second and one of the most crucial ideas here, is that in order to mimic a target distribution, we can use another function approximaator which tries to differentiate between samples generated from the true target distrubition and samples generated from the neural network trying to mimic the target distribution. Both of them will try to get better and better at their task and therefore learn by doing this. 

We implement this idea in multiple ways : 

1. Standard GANs : here the entire idea is that the discriminator will try to assign a probability for every generated and true sample and it will learn how to do this better. The assigned probability is a measure of if the said sample is true or not. The generator network will try to beat the discriminator network and therefore will try to learn how to fool it better or rather be motivated to learn in a way that the discriminator assigns the samples the generator generates a higher probability of being real. 

2. Wasserstein GANs : Wasserstein GANs tries to do what standard GANs do, but better, in the sense that in the sense that it tries to model the distributions by not considering the probability of being real or fake, but by trying to reduce the distance of two distributions (defined as the earth mover's distance). The training idea remains the same. 

3. Energy based GANs : here the entire idea is that the discriminator will try relate a embedding (generator by the discrminator) for both a generated and real sample. Furthermore, the discriminator tries to increase the distance between such embedding, but the generator tries to reduce this distance by trying to fool the discriminator. The idea here therefore is that, given an image, it is encoded into a latent representation and decoded to give an image (this network is entirely convolutional). Going ahead from there, the discriminator calculates the Mean squared reconstruction error. This reconstruction error gives the training loss, since the discriminator tries to minimize this for the real images and maximises it for the fake images. Therefore implying, that the only way the generator can learn is implicitly learning to create images the discriminator is able to reconstruct as good as the real iamges. This idea is stronger than GANs and WGANs, but larger too. Generally, it has been noticed that EBGANs generally show good properties (representing the images) in their latent layers in the Autoencoder network (the discriminator)

## Experimental setup 
THe training experiment is done to create color or black and white images using these setups. 

### Running the code :
``` python <filename> learning_rate_1 learning_rate_2 momentum <folder>```

The code runs and saves the generated image after a couple of epochs in the folder name specified. Moreover, learning rate 1 pertain to the learning rate of the generator, learning rate 2 that of the discriminator and the momentum is the momentun for both of the learning systems. 

### Results : 
See the following blog posts for results, discussion and more :

1. [Deep Learning #2 (GANs)](https://medium.com/@prannaykhosla/writing-a-deep-learning-repo-2-c4589fb169b1)
2. [Deep Learning #3 (Wasserstein GANs)](https://medium.com/@prannaykhosla/writing-a-deep-learning-repo-3-c4c950b20b92)