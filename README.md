# Autonomous-Vehicle-Motion-Predicition

For one of my final projects at Penn, I built a motion prediction algorithm to predict vehicle trajectories. We implemented and tested three deep learning algorithms:

* A convolutional neural network with an XCeption71 backbone that takes in rasterized birds-eye-view images to predict trajectories
* An implementation of [Convolutional Social Pooling](https://arxiv.org/abs/1805.06771), which seeks to model vehicle-to-vehicle interactions in order to predict trajectories
* A combination of the two in the network we called "SuperNet", where inputs from both networks were combined to output a prediction with greater accuracy.


Some of the results of our system are below. Blue lines are our predicted trajectory; pink lines are the ground truth.

![Output1](./outputs/SuperNetV2Output2.png.jpg?raw=true "Title")
![Output2](./outputs/SuperNetV2Output1.png.jpg?raw=true "Title")

