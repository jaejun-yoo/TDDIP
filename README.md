# TDDIP
Time-Dependent Deep Image Prior for Dynamic MRI

## How to run 

### Getting started with Docker

```
# Pull docker environment
docker pull jaejun2004/dip-dynamicmri
# Run main.py
python main.py --PLOT
```
### Getting started with pip 

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

