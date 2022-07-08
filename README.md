# COVID19 Detection Classifier Recipe

This pytorch recipe aims to train classifiers in pytorch to detect COVID19 from X-ray images. The dataset used to train and evaluate models is the [Chest X-ray Image Dataset](https://github.com/RishitToteja/Chext-X-ray-Images-Data-Set). This recipe provides training for multiple architectures

* Vanilla CNN (CNN layers + Dense Layers)
* [CovidNet](https://github.com/iliasprc/COVIDNet)
* ImageNet Pretrained model from torchvision models referenced [here](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)

The classifier is trained to identify an input x-ray image as COVID19 patient or normal patient.

## Structure

* [Installation](#installation)
* [Dataset](#dataset)
  * [Preprocessing](#preprocessing)
* [Training](#training)
  * [Model Configuration](#model-configuration)
* [Evaluation](#evaluation)
* [Inference](#inference)
* [Results](#results)

## Installation

Requirements

* Python3
* torch
* torchvision
* opencv-python
* tqdm
* numpy
* albumentations
* scikit-learn
* PyYAML
* tensorboard
* tensorboardX

Install python then install the other dependencies using `pip install -r requirements.txt`. For training the target device should have an nvidia GPU and cuda installed.

## Dataset

The dataset used here is the [Chest X-ray Image Dataset](https://github.com/RishitToteja/Chext-X-ray-Images-Data-Set). The data is a collection of grayscale x-ray images of variable sizes. The dataset is organized as two splits `train` and `test`. In my experiments I use `train` split for training and `test` as evaluation. The dataset directory hierachy is as follows

```text
Dataset/Data
    test/
        COVID19/
        NORMAL/
    train/
        COVID19/
        NORMAL/
```

The dataset objects for training and evaluation expect the data to be in a similar style. When you are required to enter a path to a dataset it is expected to have the following structure

```text
dataset_root/
    class1/
        images...
    class2/
        images...
    ...
```

The training pipeline is setup to handle any number of classes.

### Preprocessing

Preprocssing is setup as part of the dataset and is defined by the `dataset` config section in the configuration file.

Preprocessing steps for training are

* [`CLAHE (Contrast Limited Adaptive Histogram Equalization)`](https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.transforms.CLAHE): used to adjust input image contrast since x-ray images only have limited set of colors. Used only if the paramtere `use_CLAHE` is true.
* Resizing the image to a fixed size defined by `img_size` in configuration file
* Cropping a patch of size `crop_size` from a random location. This is the model input.
* Randomly flipping the image horizontally with probability of 25%

Preprocessing during evaluation, inference

* Using CLAHE if enabled
* Resizing to `img_size`
* Center cropping to `crop_size`

## Training

To train a classifier model use the script `trainer.py` to run training. To select model parameters, dataset configuration and training parameters you should write a conf file in yaml and provide it to the script as one of its arguments. The typical script usage is

```text
python trainer.py --train-dataset <path-to-train-dataset> --dev-dataset <path-to-devset> 
                  --save-dir <path-to-save-checpoints> --log-dir <path-for-tensorboard-logs>
                  --conf <path-to-conf-file> --class-map <path-to-class-map>
```

Parameters

* `--train-dataset`: path to the train split root directory
* `--dev-dataset`: path to the dev split root directory
* `--save-dir`: path to directory where model and optimizer checkpoints are saved
* `--log-dir`: path to save tensorboard logs
* `--conf`: trainer conf path
* `--class-map`: class map yaml file

The class map file is a simple yaml file that contains the class label to its index in the model prediction output. This is an example of a class map file

```yaml
NORMAL: 0
COVID19: 1
```

The configuration file specifes model parameters, training parameters and dataset parameters, the typical config file is defined as follows

```yaml
model:              # Model configration
  type: cnn         # Model type to select for training/evaluation
  conf:             # Model specific configuration changes from model to model according to type
    inchn: 3
    conv: [
      [16,7,7,2,2,3,3,'bn','relu','',3,3,2,2,1,1],
      [32,5,5,2,2,2,2,'bn','relu','',3,3,2,2,1,1],
      [32,5,5,2,2,2,2,'bn','relu','',3,3,2,2,1,1],
      [64,3,3,2,2,1,1,'bn','relu','',3,3,2,2,1,1],
      [128,3,3,2,2,1,1,'bn','relu','',3,3,2,2,1,1]
    ]
    dense: [
      [16, 'relu'],
      [2, '']
    ]

dataset:                        # Dataset configuration
  img_size: 256                 # Input image is resized to the fixed size 256*256
  crop_size: 224                # Input tensor to the model is a crop of size 224*224
  img_affixs: [png,jpg,jpeg]    # Supported image extensions when building dataset objects
  use_CLAHE: True               # Use CLAHE on images before resizing and cropping
  
trainer:                        # Trainer paremeters
  model_affix: CNN5x            # Name to use when saving the model
  optim_affix: Adam             # Name to use when saving optimizer state
  optim: Adam                   # Optimizer type (Adam|SGD)
  lr: 0.001                     # Learning rate
  scheduler: plateau            # Scheduler type (plateau|linear|step)
  schd_params:                  # Scheduler parameters according to their definition in pytorch
    factor: 0.1
  batch_size: 64                # Batch size
  start_epoch: 0                # Start epoch if greater than zero training will resume from this epoch
  n_epochs: 60                  # Number of total training epochs
  n_workers: 8                  # Number of workers for data loader
  log_interval: 100             # How many iterations between each logged scalars
  save_interval: 10             # Epochs between any two consecutive checkpoints
```

To start an experiment you should start by

* Putting the data in the format specified in [Dataset](#dataset)
* Write the configuration file and configure the model parameters
* run trainer.py

All the configuration are specified above except for the model configuration part since it depends on the target model. 

### Model Configuration

You can see the model configuration [here](models/README.md).

## Evaluation

To evaluate a trained model use the evaluation script `eval_dataset.py`

```text
python eval_dataset.py --device [cuda|cpu] --dataset-type [dev|test] 
                       --cls-map <class-map> --conf <conf-file> 
                       --ckpt <model-ckpt> [--print-report] 
                       --dataset <dataset-location> --result-dir <result-dir-path>
```

The class map and conf file are the same used in training.

* `--print-report`: is an optional flag that forces the script to print an sklearn classification report to stdout
* `--dataset-type`: determines what kind of dataset to expect. `dev` means that the data is divided into directories as specified in [Dataset](#dataset) where each directory represents a class. `test` means that all data is unlabled and exist under the same directory with no hierarchy.

If `dev` dataset is used the script outputs the following in the result dir

* `preds.csv`: CSV file of predictions. Image file name against its label
* `conf_mat.npy`: npy saved confusion matrix
* `cls_report.json`: Classification report in JSON format (this file is generated if `--print-report` is not specified)

## Inference

To run direct inference on images use `inference.py`

```text
python inference.py --mode [single|list] --device [cuda|cpu]
                    --conf <conf-file> --ckpt <ckpt>
                    --cls-map <class-map> <input>
```

* `--mode`: Defines to treat the `<input>` argument. `single` means the input is just a path to an image. `list` means that it is a file where a path to an image is on a new line like so

  ```text
  img1_path
  img2_path
  ...
  ```

The rest of the arguments are the same as evaluation script. The output of the script is printed to stdout and it is the predicitions in this csv format.

## Results

The following are some experiment results reported on the test split from chest x-ray dataset (484 samples). All reported results are at epoch 60 checkpoint and using weighted average since classes aren't balanced in dataset. For more detailed look at the result see [this google doc](https://docs.google.com/document/d/1pNZspvqSzd4umzy-m3bM47earQ2-1gnM3N5M0SWUVg4)

|  Model | Using CLAHE | Number of parameters | Accuracy | Precision | Recall | F1 score |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|
|CNN5x (5 CNN layers + 2 Dense)| yes | 135.83K | 100% | 100% | 100% | 100% |
|CNN5x (5 CNN layers + 2 Dense)| no | 135.83K | 99.17% | 99.17% | 99.17% | 99.17% |
|COVIDNet small| yes | 11.94 M | 98.87% | 98.97% | 98.87% | 98.96% |
|COVIDNet small| no | 11.94 M | 99.79% | 99.84% | 99.70% | 99.77% |
|Efficientnet_b0 (Imagenet Pretrained)| yes | 4.01 M | 99.79% | 99.79% |99.79% | 99.79% |
|Efficientnet_b0 (Imagenet Pretrained)| no | 4.01 M | 99.79% | 99.79% | 99.79% | 99.79%|
|ResNet50 (Imagenet Pretrained)| yes | 23.51 M | 100% | 100% | 100% | 100% |
|ResNext50 (Imagenet Pretrained| yes | 22.98 M | 100% | 100% | 100% | 100% |

From these results we can conclude

* That for this dataset a simple CNN classifier with no complexity what so ever can get performance comparable to more complex models
* CLAHE use helps vanilla model performance a little bit, but doesn't contribute to the performance of more complex models.
* Maybe more data is needed for better generalization
