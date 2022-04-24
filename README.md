<div id="top"></div>

# TPSNet: A Memory Efficent ResNet for Image Classification
By [Sudharsan Ananth](https://sudharsanananth.wixsite.com/sudharsan), [Taiyi Pan](https://www.linkedin.com/in/taiyipan/), [Pratyaksh Prabhav Rao](https://praopratyaksh1.wixsite.com/pratyaksh10),

New York University

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About This Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#methodology">Methodology</a></li>
    <li><a href="#experiment">Experiment</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

![Product Name Screen Shot](/images/Final_ResNet_model_architecture.png)

Recent advancements in deep learning have led to highly accurate image classification models. Deep Convolutional neural networks intergrate feature extractors (series of convolutional layers) and classifiers in an end-to-end fashion. Network depth plays a vital role in model accuracy, as the levels of features can be enriched by stacking more convolutional layers. However, it is observed that for deep networks, the model suffers from the vanishing/exploding gradients and the degradations problem.

ResNet model has paved the way to train deeper neural network efficiently. This project explain how we managed to acheive higher accuracy with the same ResNet architecture and the methodology used to optimize the ResNet model. The hyperparameters that make the major impact in the model are also outlined. 

The objective of this project:
* Create a model and train it on Cifar-10 dataset to get highest test accuracy. 
* The model should not exceed 5 Million Parameters to be used in mobile applications.
* Test Different Methodology to optimize the model parameters.
* Test Various Techniques to optimize the hyperparameters. 
* Conclude result by comparing differernt models.

This Project showed interesting results and we think would help community understand ResNet models better. This research is built on top of ResNet model architecture. 

Use the `README.md` to get started.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

This project is built with the below given major frameworks and libraries. The code is primarily based on python. Some plugins (Weights and Biasis and Matplotlib) are added for easy visualization of the work (and is not needed for training).

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Weights & Biases](https://wandb.ai/site)
* [Torch Ray Tune](https://docs.ray.io/en/latest/tune/getting-started.html)
* [Matplotlib](https://matplotlib.org/)
* [Cifar-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To reproduce the results and to download the model follow the procedure in this section. 

### Prerequisites

This project uses python. Hence python version 3.7.xx or higher should be installed. We recommend installing with Anaconda and adding it to the path of Windows. Git is also recommended to easily download files from GitHub. Alternatively you can also download the folder from GitHub directly. The links for both are given below
* [Python](https://www.python.org/)
* [Git](https://git-scm.com/)

### Installation

_How to reproduce the result and or clone the repository_

1. Clone the repo
   ```sh
   git clone https://github.com/taiyipan/TPSNet
   ```
2. Change Directory into the folder
   ```
   cd TPSNet
   ```
3. Install requirements
   ```sh
   pip3 install requirements.txt
   ```
4. Run train script `train.py` to recreate similar model
   ```sh
   python3 train.py
   ```
5. To Reproduce the accuracy of TPSNet(model), run `test.py` and ensure the model is on the right folder. This script will normalise the images to right value.
   ```sh
   python3 test.py
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

## Methodology

Residual blocks can be defined as a function 
xl+1 = xl + F (xl, Wl) (1).
where, xl+1 and xl are input and output of the l-th layer of the ResNet network, F is the residual function and W are the block parameters. The ResNet consists of N residual layers, where each layer consists of one or more residual blocks. The input to the network is a tensor Xinput with shape < Hi, Wi, Ci >, where Hi and Wi are the spatial dimensions and Ci is the number of channels. 

Our goal is to maximize test accuracy for a given resource constraint. The model parameters we are allowed to change are - 
1. Number of residual layers, 
2. Number of blocks in residual layer i, 
3. Number of channels in residual layer i 
4. Convolutional kernel size in residual layer i, 
5. Skip connection kernel size in residual layer i, and 
6. Average pool kernel size.

In this project, we adopt the random search technique for hyperparameter tuning. The main goal of our learning algorithm is to find a function fθ that minimizes the pre-defined loss L(x; fθ ) over a dataset Xtrain. The learning algorithm computes f through the optimization of a set of parameters θ. The random search technique involves defining a search space as a bounded domain for the model hyperparameters and randomly sample points in that domain.

![Random Search](/images/num_block_inplane_effects.png)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Experiment

The proposed architecture was evaluated on the CIFAR-10 dataset. The dataset consists of 32x32 color images drawn from 10 different classes. The CIFAR-10 dataset consists of 50,000 train and 10,000 test images. The model was implemented with PyTorch. 
![hyperparamer tuning](/images/hyperparameters_effects.png)
The network parameters are optimized using stochastic gradient descent (SGD) with an initial learning rate of 0.01 and a cosine annealing learning rate scheduler. This scheduling technique starts off with a large learning rate and then aggressively decreases it to a value near 0, before again increasing the learning rate. The loss function used is cross entropy. These parameters are found to be the best from the result of random search. 

<p align="right">(<a href="#top">back to top</a>)</p>

## Results

From various testing we have taken the best parameters for top 3 models. And we have trained for 300 epochs. 

|Baseline Models (N)| No of Residual blocks in each layer | No of channels in first conv layer (C1) | No of params | Test Accuracy(%) |
| -----------  | -----------           | --------------        | ------ | ----------- |
| Model 1      | [3,3,3]               | 34                    | 4.87M  | 94.02%      |
| Model 2      | [7,7,7]               | 22                    | 4.97M  | 94.24%      |
| TPSNet       | [13,13,13]            | 16                    | 4.96M  | 94.84%      |

The TPSNet model we trained has acheived an accuracy of 94.84%. This model is trained with optimal parameters and is a iteration of hundreds of models. The results shows the deeper ResNet models are superior in performance and generalises well without overfitting. 


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

[Taiyi Pan]([taiyipan@gmail.com](https://www.linkedin.com/in/taiyipan/)) - taiyipan@gmail.com

[Pratyaksh Prabhav Rao]([pr2257@nyu.edu](https://sudharsanananth.wixsite.com/sudharsan)) - pr2257@nyu.edu


[Sudharsan Ananth]([sudharsan.ananth@gmail.com](https://sudharsanananth.wixsite.com/sudharsan)) - sudharsan.ananth@gmail.com


Project Link: [https://github.com/taiyipan/TPSNet](https://github.com/taiyipan/TPSNet)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We would like to express our thanks to the people who's discussion helped us through the project. We are grateful to Prof. Siddharth Garg, Prof. Arsalan Mosenia and the teaching assistant Ezgi Ozyilkan for their nonstop support. Lastly, we would like to extend our special thanks to the  teaching team for giving us this opportunity to work on these assignments and projects. They were extremely helpful and pertinent to understanding the concepts.

[Siddharth Garg](https://engineering.nyu.edu/faculty/siddharth-garg?msclkid=47fb5f5dabed11ecbf387043e42d37ff)

[Arsalan Mosenia](http://www.arsalanmosenia.com/?msclkid=666ff071abed11ec93028e78876e48a0)

[Ezgi Ozyilkan](https://github.com/ezgimez?msclkid=7ab30b9dabed11ecbc97e757df7aeb95)

<p align="right">(<a href="#top">back to top</a>)</p>

