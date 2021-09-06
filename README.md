# TD-DIP (IEEE Transactions on Medical Imaging, 2021)
The Official PyTorch Implementation of __Time-Dependent Deep Image Prior for Dynamic MRI__ ([Journal](https://ieeexplore.ieee.org/document/9442767) | [arXiv](https://arxiv.org/abs/1910.01684))

Jaejun Yoo<sup>1</sup>, Kyong Hwan Jin<sup>1,2</sup>, Harshit Gupta<sup>1</sup>, Jerome Yerly<sup>2,3,4</sup>, Matthias Stuber<sup>2,3,4</sup>, Michael Unser<sup>1</sup>

<sub>**Affiliations:**</sub> <sup>1</sup> <sub>EPFL,</sub>
<sup>2</sup> <sub>DGIST,</sub>
<sup>3</sup> <sub>CHUV,</sub>
<sup>4</sup> <sub>UNIL,</sub>
<sup>5</sup> <sub>CIBM</sub>

We propose a novel unsupervised deep-learning-based algorithm for dynamic magnetic resonance imaging (MRI) reconstruction. Dynamic MRI requires rapid data acquisition for the study of moving organs such as the heart. We introduce a generalized version of the deep-image-prior approach, which optimizes the weights of a reconstruction network to fit a sequence of sparsely acquired dynamic MRI measurements. Our method needs neither prior training nor additional data. 
In particular, for cardiac images, it does not require the marking of heartbeats or the reordering of spokes.  The key ingredients of our method are threefold: 1) a fixed low-dimensional manifold that encodes the temporal variations of images; 2) a network that maps the manifold into a more expressive latent space; and 3) a convolutional neural network that generates a dynamic series of MRI images from the latent variables and that favors their consistency with the measurements in _k_-space. Our method outperforms the state-of-the-art methods quantitatively and qualitatively in both retrospective and real fetal cardiac datasets. 
To the best of our knowledge, this is the first unsupervised deep-learning-based method that can reconstruct the continuous variation of dynamic MRI sequences with high spatial resolution. 

## Updates
* 07 September 2021: Updated the nufft codebase repo 
    * **note** <s>git clone https://github.com/marchdf/python-nufft.git</s> <-- this (which I used for my Journal work) is now not available please use the link I put below.
* 20 May 2021: Paper accepted at IEEE TMI.
* September 2019: Paper submitted to IEEE TMI.

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
git clone https://github.com/dfm/python-nufft
cd python-nufft 
python setup.py install 
cd .. 
```

Now you should be able to run the code: 

```
python main.py
```

If you want to resume your training from epoch 0: 

```
python main.py --isresume ./logs/retro_YMDHMS/0.pt
```

If you want to test the model with weights saved at epoch 0: 

```
python main.py --istest --isresume ./logs/retro_YMDHMS/0.pt
```

#### Getting started with Docker

```
# Pull docker environment
docker pull jaejun2004/dip-dynamicmri
# Run main.py
python main.py
```

## Datasets
Currently we cannot open the dataset. Please contact jaejun.yoo88@gmail.com

## License
All material, excluding the dataset, is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license by Jaejun Yoo. You can use, redistribute, and adapt the material for non-commercial purposes, as long as you give appropriate credit by citing our paper and indicating any changes that you've made.

## Citation
If you find this work useful for your research, please cite:
```
@article{yoo2021time,
  title={Time-Dependent Deep Image Prior for Dynamic MRI},
  author={Yoo, Jaejun and Jin, Kyong Hwan and Gupta, Harshit and Yerly, Jerome and Stuber, Matthias and Unser, Michael},
  journal={IEEE Transactions on Medical Imaging},
  year={2021},
  publisher={IEEE}
}
```

## Contact
Feel free to contact me if there is any question (Jaejun Yoo jaejun.yoo88@gmail.com).
