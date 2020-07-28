# Crossbar Compatible ResNet Training

The codes are adapted from [the official keras example](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)

Two techniques are adopted to improve the inference accuracy on crossbar: (1) Gaussian noise injection into weights, (2) trained activation quantization. The custom Keras layers implementing the techniques are in [layer_utils.py](layer_utils.py). The Resnet models that use the custom layers are in [resnet_model.py](resnet_model.py).

To train a model:
```
python cifar10_resnet.py 3 0.2 1 3
```
where the first argument is the bit precision of the activations and the inputs; the second argument is the strength of the injected Gaussian noise (relative to the maximum absolute value of the weights of that layer); the thrid argument (optional) is the version of ResNet and the forth argument (optional) is the depth of ResNet.

To test the model's resiliency to different weight noise strength, use the [CIFAR10_CNN.ipynb](CIFAR10_CNN.ipynb).
