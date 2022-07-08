# Model Configuration

We have three main types of models to use in this recipe (the type values are what you write in the config file under `model.type`)

* `cnn`: Vanilla CNN with architecture configurable by the user
* `covid_net`: Covid Net according to [this implementation](https://github.com/iliasprc/COVIDNet)
* `pretrained`: Imagenet pretrained models from torchvision models

## Vanilla CNN

(type :`cnn`): Is a sequential series of CNN layers followed by a sequential series of dense layers. The architecture is defined in the configuration file but it's mainly as such

```text
CONV2D Stack
Global Average pooling
Dense Stack
```

An example configuration

```yaml
model:
type: cnn
conf:
inchn: 3                                            # Number of input channels
conv: [                                             # Start defining conv layers
    [16,7,7,2,2,3,3,'bn','relu','',3,3,2,2,1,1],      # One Layer definition
    [32,5,5,2,2,2,2,'bn','relu','',3,3,2,2,1,1],
    [32,5,5,2,2,2,2,'bn','relu','',3,3,2,2,1,1],
    [64,3,3,2,2,1,1,'bn','relu','',3,3,2,2,1,1],
    [128,3,3,2,2,1,1,'bn','relu','',3,3,2,2,1,1]
]
dense: [                                            # Start defining dense layers
    [16, 'relu'],                                     # one layer definition
    [2, '']
]
```

The conv layer definition is a list of values in the following structure

```text
[output channels, k1, k2, s1, s2, p1, p2, norm method, activation, pooling, kp1, kp2, sp1, sp2, pp1, pp2]
```

* `output channels`: The number of output channels from conv2d
* `(k1,k2)`: conv2d Kernel size
* `(s1,s2)`: conv2d Stride
* `(p1,p2)`: conv2d padding
* `norm method`: nomralization method to use before activatation.Allowed values
* `bn` Batch normalization
* `ins` Instance normalization
* `activation`: activation function. Allowed values
* `relu` ReLU activation
* `prelu` Parametric ReLU activation
* `leakyrelu` Leaky ReLU activation
* `pooling`: The type of pooling to use after activation Allowed values
* `max`: Maxpooling
* `avg`: Average pooling
* `(kp1, kp2)`: Pooling layer kernel size
* `(sp1, sp2)`: Pooling layer stride
* `(pp1, pp2)`: Pooling layer padding

You can change the architecture by specifying any number of layers you want and by writing any parameters you want. You don't have to specify all parameters if you don't wish to use a module simply write empty string in its location. For example if you don't want to use normalization just write `''` in its place. This is an example for just conv2d with nothing else

```yaml
[32,5,5,2,2,2,2,'','','',1,1,1,1,1,1]
```

Note that despite not using pooling you must supply any six numbers.

## Covid Net

Covid net is convolutional network based on the paper [COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images](https://www.nature.com/articles/s41598-020-76550-z). The implementation used here is based on a pytorch implementation of this architecture. You can find it [here](https://github.com/iliasprc/COVIDNet).

The configuration for this network is pretty straight forward, we simply specify the model type `covid_net` and the architecture which has one of two values `small` and `large`. This recipe supports only small COVIDNet architecture.

Example configuration

```yaml
model:
  type: covid_net
  conf:
    model: small
    n_classes: 2
```

* `model.conf.model` specifies architecture type
* `model.conf.n_classes` specifies the number of output classes

## ImageNet Pretrained

This recipe also allows using pretrained models from torchvision models (which are pretrained on imagenet) and finetune them using the dataset. You can find a list of all pretrained models in [torchvision documentation](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights). This recipe supports only

* `resnet18`
* `resnet50`
* `resnext50`
* `efficientnet_b0`

Others can be added easily to the [`imagenet_pretrained.py`](models/imagenet_pretrained.py) script. To specify which model to use simply specify the `model.type` as `pretrained` and the `model.conf.model_type` to be one of the above values.

An example configuration

```yaml
model:
  type: pretrained
  conf: 
    model_type: resnet50
    n_cls: 2
```
