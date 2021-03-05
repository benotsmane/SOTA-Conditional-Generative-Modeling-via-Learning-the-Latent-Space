This directory contains scripts for three experiments described in the paper Conditional Generative Modeling via Learning the Latent Space
by Ramasinghe et al ICLR 2021.
Master DAC Students :
    - Marc Lafon
    - Ismat Benotsmane
    - Tianwei Lan

Modele.py:
Contains the model architecture for MNIST experience


Experience_1.py:
We alter 20% of the data during train by filling the top half of an image with an other random image's top half.
The purpose here is to observe that our model is not affected by this alteration and still generate very good results compared to other models.

Experience_2.py:
We added a white line to 30% of the data during train, to have 2 mode in the dataset. One mode for the original data, and the other mode for the data with a white line.
The purpose is to observe that depending on the initialisation of z, we may have a mode_1 output as the regular image or a mode_2 output as an image with a white line.


toy.py:
Contains the training and graph generation code for the toy experiment which consist of a bi-modal continuous regression problem : y= +-4 (x, x**2, x**3)

networks_toy.py
Contains networks architectures for the toy experiment

utils.py
Contains utilitary functions