# ENet-ScanNet

PyTorch (v1.0.0) re-implementation of [*ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*](https://arxiv.org/abs/1606.02147), ported from the excellent PyTorch impelentation [ENet-ScanNet](https://github.com/davidtvs/PyTorch-ENet), which was in-turn ported from the lua-torch implementation [ENet-training](https://github.com/e-lab/ENet-training) created by the authors.

This implementation has been made with the [ScanNet](http://www.scan-net.org) dataset in mind.


## Installation

1. Python 3 and pip.
2. Set up a virtual environment (optional, but recommended).
3. Install dependencies using pip: ``pip install -r requirements.txt``.


## Usage

Run [``main.py``](https://github.com/krrish94/ENet-ScanNet/blob/master/main.py), the main script file used for training and/or testing the model. The following options are supported:

```
python main.py [-h] [--mode {train,test,full}] [--resume]
               [--batch-size BATCH_SIZE] [--epochs EPOCHS]
               [--learning-rate LEARNING_RATE] [--lr-decay LR_DECAY]
               [--lr-decay-epochs LR_DECAY_EPOCHS]
               [--weight-decay WEIGHT_DECAY] [--dataset {camvid,cityscapes}]
               [--dataset-dir DATASET_DIR] [--height HEIGHT] [--width WIDTH]
               [--weighing {enet,mfb,none}] [--with-unlabeled]
               [--workers WORKERS] [--print-step] [--imshow-batch]
               [--device DEVICE] [--name NAME] [--save-dir SAVE_DIR]
```

For help on the optional arguments run: ``python main.py -h``


### Examples: Training

```
python main.py -m train --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


### Examples: Resuming training

```
python main.py -m train --resume True --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


### Examples: Testing

```
python main.py -m test --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


## Project structure

### Folders

- [``data``](https://github.com/krrish94/ENet-ScanNet/tree/master/data): Contains instructions on how to download the datasets and the code that handles data loading.
- [``metric``](https://github.com/krrish94/ENet-ScanNet/tree/master/metric): Evaluation-related metrics.
- [``models``](https://github.com/krrish94/ENet-ScanNet/tree/master/models): ENet model definition.
- [``save``](https://github.com/krrish94/ENet-ScanNet/tree/master/save): By default, ``main.py`` will save models in this folder. The pre-trained models can also be found here.

### Files

- [``args.py``](https://github.com/krrish94/ENet-ScanNet/blob/master/args.py): Contains all command-line options.
- [``main.py``](https://github.com/krrish94/ENet-ScanNet/blob/master/main.py): Main script file used for training and/or testing the model.
- [``test.py``](https://github.com/krrish94/ENet-ScanNet/blob/master/test.py): Defines the ``Test`` class which is responsible for testing the model.
- [``train.py``](https://github.com/krrish94/ENet-ScanNet/blob/master/train.py): Defines the ``Train`` class which is responsible for training the model.
- [``transforms.py``](https://github.com/krrish94/ENet-ScanNet/blob/master/transforms.py): Defines image transformations to convert an RGB image encoding classes to a ``torch.LongTensor`` and vice versa.
