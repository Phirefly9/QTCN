# Code For Quaternion Temporal Convolutional Networks

This repo is the cifar10/100 code for [Quaternion Temporal Convolutional Networks](https://etd.ohiolink.edu/pg_10?::NO:10:P10_ETD_SUBID:181679)


## Code dependencies

The repo requires Nvidia's apex library so I recommend running the code in [Nvidia's pytorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch), which has all of the required dependencies installed.

If you want to install the dependencies I have provided a conda requirements yml

```
conda create -yn QTCN
conda env update -f requirements.yml
```

Apex cannot be installed by conda, so you will have to install it manually.

## Running the code

in the main repo directory
```
python -m QTCN.main
```

## Code Acknowledgements 

This repo is heavily based on code from the following three repos:

1. https://github.com/locuslab/TCN for the base TCN code
2. https://github.com/icpm/pytorch-cifar10 for the base Cifar solver
3. https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks for the base Quaternion convolution and initialization code

## License

The premodified functions the QTCN.qtcn.quaternion_layers and quaternion_ops files are GPL'd under https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks, so the entire repo must be GPLV3.

The file header for the base quaternion_layers and quaternion_ops file occurs as follows:

```
##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################
```


## Citing

If this code helps at all please consider citing it using

```
@phdthesis{long2019quaternion,
  title={Quaternion Temporal Convolutional Neural Networks},
  author={Long, Cameron E},
  year={2019},
  school={University of Dayton}
}
```