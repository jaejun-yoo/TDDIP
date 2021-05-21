# TD-DIP (IEEE Transactions on Medical Imaging, 2021)
The Official PyTorch Implementation of [Time-Dependent Deep Image Prior for Dynamic MRI](https://arxiv.org/abs/1910.01684)

Jaejun Yoo, Kyong Hwan Jin, Harshit Gupta, Jerome Yerly, Matthias Stuber, Michael Unser

We propose a novel unsupervised deep-learning-based algorithm for dynamic magnetic resonance imaging (MRI) reconstruction. Dynamic MRI requires rapid data acquisition for the study of moving organs such as the heart. We introduce a generalized version of the deep-image-prior approach, which optimizes the weights of a reconstruction network to fit a sequence of sparsely acquired dynamic MRI measurements. Our method needs neither prior training nor additional data. 
In particular, for cardiac images, it does not require the marking of heartbeats or the reordering of spokes.  The key ingredients of our method are threefold: 1) a fixed low-dimensional manifold that encodes the temporal variations of images; 2) a network that maps the manifold into a more expressive latent space; and 3) a convolutional neural network that generates a dynamic series of MRI images from the latent variables and that favors their consistency with the measurements in _k_-space. Our method outperforms the state-of-the-art methods quantitatively and qualitatively in both retrospective and real fetal cardiac datasets. 
To the best of our knowledge, this is the first unsupervised deep-learning-based method that can reconstruct the continuous variation of dynamic MRI sequences with high spatial resolution. 

## Updates
* 20 May 2021: Paper accepted at IEEE TMI 2021.

## Getting Started

### Installation

#### Getting started with pip 

If you don't have docker, but you have python3, pip3, and git, run the following commands:

```
#download virtualenv with pip3: 
pip3 install virtualenv

#start a virtual environment
virtualenv venv
source venv/bin/activate

#make sure that you are using python 3
which python 

#use the provided requirements file 
pip3 install -r requirements.txt

#clone the python-nufft library 
git clone https://github.com/marchdf/python-nufft.git
cd python-nufft 
python setup.py install 
cd .. 
```

Now you should be able to run the code: 

```
python main.py --PLOT
```
#### Getting started with Docker

```
# Pull docker environment
docker pull jaejun2004/dip-dynamicmri
# Run main.py
python main.py --PLOT
```

### Datasets
Please contact jaejun.yoo88@gmail.com

## Citation
If you find this work useful for your research, please cite (will be updated soon):
```
@article{Yoo2019time,
  title={Time-dependent deep image prior for dynamic MRI},
  author={Yoo, Jaejun and Jin, Kyong Hwan and Gupta, Harshit and Yerly, Jerome and Stuber, Matthias and Unser, Michael},
  journal={arXiv preprint arXiv:1910.01684},
  year={2019}
}
```

## Contact
Feel free to contact me if there is any question (Jaejun Yoo jaejun.yoo88@gmail.com).
