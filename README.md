# Mentis JS Machine Learning/Evolution <img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/logo.png?raw=true" width="100px">

Mentis JS is a JavaScript library for machine learning and deep learning. It provides functionalities for training neural networks using backpropagation and neural evolution. Alternative to ConvNetJs

## Online Demos

Check out our online demos [here](https://trevorblythe.github.io/MentisJS/).

## Installation

The entire library is contained within the "Mentis.js" file.

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

Networks ONLY TAKE SINGLE DIMENSIONAL ARRAYS. No more hassle with array dimensions. Inputting images is described a couple scrolls down this page.

```javascript
let output = network.forward(inputs[0]); //returns the output (an array with 1 number)
```

<p>Since our network isnt trained yet, it is useless</p>

## Training

There are two ways to train. the "train" function. or you can forward data and then backward data manually.

#### Method 1 of training

```javascript
let loss = network.train(inputArray, expectedOutputArray);
let output = network.outData; //Not a copy
```

#### Method 2 of training

```javascript
let output = network.forward(inputArray);
let loss = network.backward(expectedOutputArray);
```

Using one method doesnt stop you from using the other.

## optimizers/training methods

Optimizers are plug and play, and you can switch them out at any time. <br>
Current Optimizers include:<br>
SGD<br>
Momentum<br>
Adam<br>
AdamW (adam with weight decay)<br>

SGDGPU (sgd but but for gpu networks)<br>
MomentumGPU (momentum but for gpu networks)<br>

### 1. Just include it when making your net.

```javascript
let network = new Ment.Net([layers]); //uses SGD optimizer (default)
let network = new Ment.Net([layers], "SGD"); // also uses SGD
let network = new Ment.Net([layers], "Momentum"); // uses momentum
let network = new Ment.Net([layers], new Ment.Momentum(0.5)); //uses momentum with 0.5 momentum constant
let network = new Ment.Net([layers], new Ment.Adam(1000)); //every 1000 param updates, momentum is reset and old gradients are forgotten
```

### How to change optimizers mid-train:

```javascript
//Assume net has SGD and we wanna switch to momentum
net.optimizer = new Momentum(0.9);
net.optimizer.netObject = net;
net.optimizer.initialize();
//thats it, it will now use momentum
```

### All layer types

```javascript
FC(inputSize, outputSize, useBias?);

AveragePool([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], stride);

MaxPool([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], stride);

MinPool([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], stride);

BiasLayer(Size); //just adds a number to each activation

ConvLayer([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], howManyFilters, stride, useBias?)
DepthWiseConvLayer([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], stride, useBias?)
ConvParamLayer([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], howManyFilters, stride, useBias?, filterInputSize, filterNetworkDepth)

DeconvLayer([outputHeight, outputWidth, outputDepth], [filterHeight, filterWidth], howManyFilters, stride, useBias?)
DepthWiseDeconvLayer([outputHeight, outputWidth, outputDepth], [filterHeight, filterWidth], howManyFilters, stride, useBias?)
DeconvParamLayer([inputHeight, inputWidth, inputDepth], [filterHeight, filterWidth], howManyFilters, stride, useBias?, filterInputSize, filterNetworkDepth)

PaddingLayer([inputHeight, inputWidth, inputDepth], padAmount, whatValueToPadWith)

DepaddingLayer([outputHeight, outputWidth, outputDepth], padAmount)

ResEmitter(identificationString) //residual emitter
ResReceiver(identificationString, mode (concat, add, average)) //residual receiver

RecEmitter(identificationString) //reccurrent emitter
RecReceiver(identificationString, mode (concat, add, average)) //reccurrent receiver

Sigmoid(Size (optional parameter for all activations));
Tanh(Size);
PTanh(Size); //periodic tanh for the haters
Relu(Size);
LeakyRelu(leakySlope, Size);
Sine(Size);
Identity(Size); //an activation that does nothing and provides no real training value

UpscalingLayer([inputHeight, inputWidth, inputDepth], scale)

LayerNormalization(size);
//very new might have errors/bugs but seems to work

```

### Dynamic Convolutional filters

Layers like ConvParamLayer and DeconvParamLayer allow you to have dynamic filters. This means that the filters are not fixed, and are instead dynamically generated by a separate small network. This allows for more complex and dynamic filters.

```javascript

var net = new Ment.Net(
	[
		//28 28 3 input size, 1 by 1 kernel size, 1 filter, 1 stride, no bias, 3 filter input size, 1 filter network depth
		new Ment.ConvParam([28, 28, 3], [1, 1], 1, 1, false, 3, 1),
		new Ment.DeConvParam([28, 28, 3], [1, 1], 1, 1, false, 3, 1),
	],
);

//Training loops with dynamic filters require special handling
for(var i = 0; i < 1000; i++){
	let input = //some array of size [28, 28, 3]
	let expected_output = //some array of size [28, 28, 3]
	let latent_input = //some array of size [3] (the filter input size)
	net.layers[0].forwardFilter(latent_input);
	net.layers[1].forwardFilter(latent_input);
	net.forward(input);
	net.backward(output);
}

```

### Inputting Images into the network

Networks can only take <b>SINGLE DIMENSIONAL</b> arrays as inputs. So to input an image, just input the image as an rgb array of pixel values.

Do it LIKE THIS

```javascript
let inputImage = [r, r, r, g, g, g, b, b, b];
```

NOT LIKE THIS

```javascript
let inputImage = [r, g, b, r, g, b, r, g, b];
```

I had to choose a format so I just chose the first one.
You can also have images with more dimensions then rgb like rgba or any amount of depth you want.

# RNN's

You can make simple RNN's with this library.

## Reccurent Layers

<b>RecEmitterLayer</b> saves its input and then forwards it to its corressponding <b>RecReceiverLayer</b> the next time you call <b>.forward()</b> on the model

<img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/rnn_diagram.png?raw=true" >

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

	net.backward([0]); //expected output zero for input [0,0]
	net.backward([0]); //expected output zero for input [1,1]
	net.backward([1]); //expected output one for input [1,0]
	net.backward([1]); //expected output one for input [0,1]
}
```

This is shown below:

```javascript
//normal training:
net.forward(input1);
net.backward(output1);
net.forward(input2);
net.backward(output2);

//---- Versus -----

//rnn training
net.forward(input1);
net.forward(input2);

net.backward(output2);
net.backward(output1);
```

## Making Rnn feed into itself

Calling "forwards" on an rnn with no parameter provided will just use its last output as its input. You can also do this with "backwards" and it will backpropagate the error from last time it was backwarded.

This will only work if the input size of the network is the same as the output size.

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

Every Layer has a GPU version. Just add "GPU" after the constructor names.
Only some are implemented in GPU though.

It uses Webgl and an outdated library called WebMonkeys. It compiles the shaders on the fly so you dont need an NVIDIA GPU to use it.

Very far from done because I dont wanna invest to much time into it but its a proof of concept that you can use the GPU whilst also having a simple API that doesn't require a specific GPU or anything complicated like that.

### How to use Mentis on gpu.

Just add "GPU" after the constructor names. Take Note you also need to use the "NetGPU" constructor instead of "Net" and the MomentumGPU optimizer instead of the normal Momentum optimizer.

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

# Backpropagating custom gradients/error

If you wanna do this, there is a parameter in the "backwards" function you can change.

Make sure to multiply your error by -1 if it doesn't work because Gradients are added to the weights and not subtracted like other libraries.

Error + ActualOutput = ExpectedValue

```javascript
//these are the parameters for the backwards function
network.backward(Expected/Grads, CalcLoss?, BackPropGradsInstead?);
```

## Getting grads/error from the network

Every Layer has its own set of grads. For cpu you can just do this.

```javascript
network.layers[0].grads;
```

# End

I think i might be done with this passion project. Open up "index.html" for lots of cool examples.
trevorblythe82@gmail.com <--- Ask me questions?

And as for PyTorch, with its terrible APIs and workflow, I choose to stick with MentisJS. Its clean and intuitive API makes deep learning development simple. No more struggling with convoluted code and confusing documentation or confusing tensor shapes or needing a specific 1000 dollar gpu when I can just use my non monopolistic 100 dollar gpu and get better results with WebGPU or webgl. You guys were great machine learning researchers, but you are terrible at making a good deep learning framework. I'm sorry, but I'm moving on.
