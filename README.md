<div id="top"></div>

# DRLEVN: Deep Reinforcement Learning Embodied Visual Navigation
By [Sudharsan Ananth](https://sudharsanananth.wixsite.com/sudharsan), [Taiyi Pan](https://www.linkedin.com/in/taiyipan/), [Pratyaksh Prabhav Rao](https://praopratyaksh1.wixsite.com/pratyaksh10),

New York University

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#About-the-project">About This Project</a>
      <ul>
        <li><a href="#Built-with">Built With</a></li>
        <li><a href="#Paper Reference">Paper Reference</a></li>
        <li><a href="#Tools used">Tools used</a></li>
      </ul>
    </li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#Step-by-Step Installation">Step-by-Step Installation</a></li>
      <ul>       
          <li><a href="#Install Anaconda and pip">Install Anaconda</a></li>
          <li><a href="#Create conda environment">Create conda environment</a></li>
          <li><a href="#Install basic packages in environment">Install basic packages in environment</a></li>
          <li><a href="#Install Habitat-sim and test">Install Habitat-sim and test</a></li>
          <li><a href="#Install Habitat Lab and Test">Install Habitat Lab and Test</a></li>
          <li><a href="#Clone SplitNet repo">Clone SplitNet repo</a></li>
          <li><a href="#Run Test">Run Test</a></li>
      </ul>
    </li>
    <li><a href="#experiments">Experiments</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project
<!--
![Product Name Screen Shot](/images/Final_ResNet_model_architecture.png)
-->
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

### Paper Reference


### Tools used


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Prerequisites -->
## Prerequisites

This project is not supported in windows. Habitat sim is not available for Windows and is available only on Mac OS and Linux. The procedure for running this experiment in Mac OS is slightly different but the steps are the same. The link for Habitat-sim is given below along with the supported OS.
* [aihabitat](https://anaconda.org/aihabitat/habitat-sim)

Also please note that these results cannot be performed in a virtual machine. The dependencies and the path conflicts and will not work in a virtual machine with any verison of Ubuntu or Linux distributions. 


## Step by Step Installation (for Ubuntu 20.04LTS)

To reproduce the results and to run the experiment follow the instructions in this section. 

### 1. Install Anaconda

1. Update Local Package Manager 
   ```sh
   sudo apt-get update
   ```
2. If your system doesn’t have curl, install it by entering:
   ```sh
   sudo apt-get install curl
   ```
3. Retrieving the Latest Version of Anaconda. Copy paste the below link in a web browser and right click the download button and copy the url
   ```sh
   https://www.anaconda.com/distribution/
   ```
4. Create a Temporary Directory, and download anaconda using curl. make sure to change the url to the one copied from the above step
   ```
   mkdir tmp
   cd /tmp
   curl –O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
   ```

5. Running the Anaconda Script. Press yes, accept the terms and aggrements and install anaconda after pasting the below line. 
   ```
   bash Anaconda3-2019.03-Linux-x86_64.sh
   ```
   
6. Activating Installation
   ```sh
   source ~/.bashrc
   ```

7. Install Pip
   ```sh
   sudo apt install python3-pip
   ```

8. Install Git
    ```sh
    sudo apt install git
    ```

### 2. Create conda environment

9. Preparing Conda Environment
    ```sh
    # We require python>=3.7 and cmake>=3.10
    conda create -n habitat python=3.7 cmake=3.14.0
    conda activate habitat
    ```
10. Installing basic package managers for easy installation. 
    ```sh
    # We need Git and pip to install requirments. Ensure to install inside the environment. 
    sudo apt install python3-pip
    sudo apt install git
    ```


<!--
To Reproduce the accuracy of TPSNet(model), run `test.py` and ensure the model is on the right folder. This script will normalise the images to right value.
   ```sh
   python3 test.py
   ```
-->
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

