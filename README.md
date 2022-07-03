# Mentis JS machine learning/evoloution <img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/logo.png?raw=true" width = 100px>

A javascript neural network library. It can train with backpropagation or neural evoloution! Ill post some examples of those soon!
I am about to be a college student so expect the next four years to be this projects prime. This project is aimed to be the most simple
and intuitive machine learning library out there. Inspired by the complicated python ML apis (to make a better one )! Also Inspired by convnetjs!

# This project is a work in progress!

Im rewriting it to be as simple as possible. It is now extremely easy to write your own layers! Look in FCLayer.js for a specification on how to write
your first one.

Things i will add:
deconvoloution -- Complete
lstm layer, soon
padding in conv layer, at 30 github stars
optimizers, soon

Things for the distant future:

GAN tutorial.
Diffusion model tutorial.
WASM for speed?

Check the wiki for more docs!

# Tutorial

<h2>How to make a Neural Network</h2>

<p>In this example we will make a neural network that will be a XOR gate.</p>
<p>First lets define the Dataset.</p>

```
let inputs = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];

let outputs = [[0], [1], [1], [0]];

```

<p>Next we must define the layers our Network will have!</p>

```
let layerOne = new Ment.FC(2,5); //input:2, output:5, fully connected layer
let layerTwo = new Ment.Sig(5); //size:5, sigmoid activation
let layerThree = new Ment.FC(5,5); //input:5, output:5, fully connected layer
let layerFour = new Ment.FC(5,1); //input:5, output:1, fully connected layer
let layerFive = new Ment.Sig(1); //size:1, sigmoid activation

```

<p>Notice how the first layer takes an input size of two? This is because our dataset input size is two!</p>
<p>The last layer only has one neuron. That will be the output neuron and will tell us what the network predicts!</p>
<p>This network is MASSIVE overkill for simple XOR but is good for a tutorial. Next we put these layers in a Net object</p>

```
let network = new Ment.Net([
  layerOne,
  layerTwo,
  layerThree,
  layerFour,
  layerFive
]);

network.learningRate = 0.1;
network.batchSize = 4;

```

<p> Now that we have our network made we can use it to solve XOR! You can input data with the forward function.</p>

```
network.forward(inputs[0]); //returns the output

```

<p>Since our network isnt trained yet, it returns gibberish... Lets train it</p>

```
	for (var i = 0; i < 1000; i++) {
		net.train(inputs[i % 4], outputs[i % 4]);
	}
```

<p>Now that it is trained lets test it</p>

```
network.forward([0,0]);
```

> [0.0001]

```
network.forward([1,0]);
```

> [0.9983]

```
network.forward([0,1]);
```

> [0.9923]

```
network.forward([1,1]);
```

> [0.0012]

# All current layers:

Conv(inDim,filterDim,filters = 3,stride = 1,bias = true)<br>
Deconv(inDim,filterDim,filters = 3,stride = 1,bias = true)<br>
Depadding(outDim, pad)<br>
FC(inSize,outSize)<br>
Identity(size)<br>
LeakyRelu(size)<br>
Relu(size)<br>
MaxPool(inDim, filterDim, stride = 1)<br>
Padding(inDim, pad, padwith)<br>
ResEmitter(id)<br>
ResReceiver(id)<br>
Sigmoid(size)<br>
Sine(size)<br>
Tanh(size)<br>

Check the wiki for more docs!
