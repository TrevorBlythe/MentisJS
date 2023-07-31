# Mentis JS machine learning/evolution <img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/logo.png?raw=true" width = 100px>

A javascript machine learning library. It can train with backpropagation and neural evolution.

# Installation

The entire library is located in "Mentis.js"

```html
<script src="Mentis.js"></script>
```

# Usage

"Ment" is the global identifier for the library. Using it you can easily make simple neural networks like this. This one takes an input of 2, and outputs 1 number.

## Making networks/models/Ai's

```javascript
let network = new Ment.Net([
	new Ment.FC(2, 5), //fully connected/dense layer input size 2 output size 5
	new Ment.Sig(5), //sigmoid activation layer size 5
	new Ment.FC(5, 5),
	new Ment.FC(5, 1),
	new Ment.Sig(1),
]);

network.learningRate = 0.1;
network.batchSize = 4;
//batch size is how many batches it will train on before averaging gradients and updating the params (weights)
```

Networks ONLY TAKE SINGLE DIMENSIONAL ARRAYS. Scroll down to learn how to input images

```javascript
let output = network.forward(inputs[0]); //returns the output (an array with 1 number)
```

<p>Since our network isnt trained yet, it is useless</p>

## Training

There are two ways to train. the "train" function. or you can forward data and then backward data manually.

#### Method 1 of training

```javascript
let loss = network.train(inputArray, expectedOutputArray);
```

#### Method 2 of training

```javascript
let output = network.forward(inputArray);
let loss = network.backward(expectedOutputArray);
```

Using one method doesnt stop you from using the other.

## optimizers/training methods

Right now there is only TWO, SGD, and momentum. Momentum is pretty much better.

### 1. Just include it when making your net.

```javascript
let network = new Ment.Net([layers]); //uses SGD optimizer (default)
let network = new Ment.Net([layers], "SGD"); // also uses SGD
let network = new Ment.Net([layers], "Momentum"); // uses momentum
let network = new Ment.Net([layers], new Ment.Momentum(0.5)); //uses momentum with 0.5 momentum constant
```

### How to change optimizers mid-train:

```javascript
//Assume net has SGD and we wanna switch to momentum
net.optimizer = new Momentum(0.9);
net.optimizer.netObject = net;
net.optimizer.initialize();
//thats it
```

### All layer types

```javascript
FC(inputSize, outputSize, useBias?);

AveragePool([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], stride);

MaxPool([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], stride);

MinPool([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], stride);

BiasLayer(Size); //just adds a number to each activation

ConvLayer([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], howManyFilters, stride, useBias?)

DeconvLayer([outputHeight, outputWidth, outputDepth], [filterHeight, filterWidth], howManyFilters, stride, useBias?)

PaddingLayer([inputHeight, inputWidth, inputDepth], padAmount, whatValueToPadWith)

DepaddingLayer([outputHeight, outputWidth, outputDepth], padAmount)

ResEmitter(identificationString) //residual emitter
ResReceiver(identificationString, mode (concat, add, average)) //residual receiver

RecEmitter(identificationString) //reccurrent emitter
RecReceiver(identificationString, mode (concat, add, average)) //reccurrent receiver

Sigmoid(Size (optional parameter));
Tanh(Size);
Relu(Size);
LeakyRelu(Size);
Sine(Size);

UpscalingLayer([inputHeight, inputWidth, inputDepth], scale)

```

### Inputting Images into the network

Networks can only take SINGLE DIMENSIONAL arrays as inputs. So to input an image, just input the image as an rgb array of pixel values.

Do it LIKE THIS

```javascript
let inputImage = [r, r, r, g, g, g, b, b, b];
```

NOT LIKE THIS

```javascript
let inputImage = [r, g, b, r, g, b, r, g, b];
```

You can also have images with more dimensions then rgb like rgba or any amount of depth you want.

# RNN's

You can make RNN's with this library.

## Reccurent Layers

<b>RecEmitterLayer</b> saves its input and then forwards it to its corressponding <b>RecReceiverLayer</b> the next time you call <b>.forward()</b> on the model

<img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/rnn_diagram.png?raw=true">

## How to use these layers?

You add them normally EXCEPT instead of Ment.Net() constructor, you have to use Ment.RNN() or it wont work.

These Layers take One parameter. Its a string. The string is how the network knows which reccurrent layers are connected to which reccurrent receivers. Make sure they are the same string in both layers.

```javascript
let net = new Ment.Rnn([
	new Ment.Input(2),
	new Ment.FC(2, 5),
	new Ment.Sig(5),
	new Ment.RecReceiverLayer("id"), //special layers here!!
	new Ment.FC(10, 5),
	new Ment.RecEmitterLayer("id"), //special layers here!!
	new Ment.FC(5, 1),
]);
```

<img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/net_diagram.png?raw=true" width = "800px">

## training Rnn's

Dont use "train" function for rnns. You have to forward the data in order, and then backwards the expected data in opposite order.

```javascript
for (var i = 0; i < 50000; i++) {
	net.forward([0, 1]);
	net.forward([1, 0]);
	net.forward([1, 1]);
	net.forward([0, 0]);

	net.backward([0]); //output zero for input [0,0]
	net.backward([0]); //output zero for input [1,1]
	net.backward([1]); //output one for input [1,0]
	net.backward([1]); //output one for input [0,1]
}
```

This is shown below:

```javascript
//normal training:
net.forward(input1);
net.backward(output1);
net.forward(input2);
net.backward(output2);

//rnn training
net.forward(input1);
net.forward(input2);

net.backward(output2);
net.backward(output1);
```

## Making Rnn feed into itself

Calling "forwards" on an rnn with no parameter provided will just use its last output as its input. You can also do this with "backwards" and it will backpropagate the error from last time it was backwarded.

```javascript
//rnn training
net.forward(input1);
net.forward();
net.forward();

net.backward();
net.backward();
net.backward(output1);
```

works.

## Clearing a models "memory"

If you wanna reset an Rnns memory/cache/activations, do this.

```javascript
net.resetRecurrentData(); // this function resets its memory
```

# GPU

Finally added gpu. Its still not 100% optimized but its pretty good and tested. Not all layers have gpu versions of themselves.

### How to use Mentis on gpu.

Just add "GPU" after the constructor names.

```javascript
let network = new Ment.NetGPU(
	[
		new Ment.FCGPU(2, 5), //fully connected/dense layer input size 2 output size 5
		new Ment.SigGPU(5), //sigmoid activation layer size 5
		new Ment.FCGPU(5, 5),
		new Ment.FCGPU(5, 1),
		new Ment.SigGPU(1),
	],
	"MomentumGPU"
);

network.learningRate = 0.1;
network.batchSize = 4;
```

### How to get outputs from the GPU

```javascript
network_on_gpu.forward(data);
//DOESNT RETURN ANYTHING

network_on_gpu.backward(data);
//ALSO DOESNT RETURN ANYTHING
```

It doesnt return anything because getting arrays from the gpu is very slow, so you should only do it very periodically to check on your networks progress. Here is how.

```javascript
network_on_gpu.forward(data);
network_on_gpu.outData;
//Returns the output array

network_on_gpu.backward(data, true);
//Returns the loss (The second parameter makes it do that)
```

Everything else is basically the same. DONT BE SUPRISED WHEN ITS SLOW ON YOUR FIRST "forward" CALL, IT IS JUST INITIALIZING THE SHADERS!! All calls after that should be very fast.

I think i might be done with this passion project. Open up "index.html" for lots of cool examples.
trevorblythe82@gmail.com
