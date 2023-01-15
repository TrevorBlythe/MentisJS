# Mentis JS machine learning/evolution <img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/logo.png?raw=true" width = 100px>

A javascript neural network library. It can train with backpropagation or neural evoloution! Ill post some examples of those soon!
I am about to be a college student so expect the next four years to be this projects prime. This project is aimed to be the most simple
and intuitive machine learning library out there. Inspired by the complicated python ML apis (to make a better one )! Also Inspired by convnetjs!

Im writing it to be as simple as possible. It is now extremely easy to write your own layers! Look in FCLayer.js for a specification on how to write
your first one.

# Tutorial

<h2>How to make a Neural Network</h2>

<p>In this example we will make a neural network that will act as a XOR gate. (a logic gate)</p>
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
let layerOne = new Ment.FC(2,5);
//input:2, output:5, fully connected layer

let layerTwo = new Ment.Sig(5);
//size:5, sigmoid activation

let layerThree = new Ment.FC(5,5);
//input:5, output:5, fully connected layer

let layerFour = new Ment.FC(5,1);
//input:5, output:1, fully connected layer

let layerFive = new Ment.Sig(1);
//size:1, sigmoid activation

```

<p>Notice how the first layer takes an input size of two? This is because we will be inputting two numbers into the network</p>
<p>The last layer only has one neuron. That will be the output neuron and will tell us what the network predicts.</p>
<p>This network is overkill for a simple XOR gate but is good for a tutorial. Next we put these layers in a Net object</p>

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

<p> It is also possible to define the layers inside of the network definition to save space like so: </p>

```
let network = new Ment.Net([
	new Ment.FC(2,5),
	new Ment.Sig(5),
	new Ment.FC(5,5),
	new Ment.FC(5,1),
	new Ment.Sig(1)
]);
```

<p> Now that we have our network made we can use it to solve XOR! You can input data with the forward function.</p>

```
network.forward(inputs[0]); //returns the output (a number)
```

<p>Since our network isnt trained yet, it returns gibberish... Lets train it</p>

```
    for (var i = 0; i < 5000; i++) {
    	net.train(inputs[i % 4], outputs[i % 4]);
    }

	//net.train() takes input and desired output as parameters.
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

It works!

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
RecReceiverLayer(id)<br>
RecEmitterLayer(id)<br>
Sigmoid(size)<br>
Sine(size)<br>
DataBlockLayer(size)<br>
Tanh(size)<br>

# RNN update

Im not completly sure if I did it correctly! This is why im releasing it with a warning that it might not work and
the api might change. Please give feedback if anyone is even using this.

A tutorial has been added to the website to get started with some rudimentary RNN models.

Im trying to get a really small diffussion model as a proof of work that my rnn implementation is working.

This latest update also comes with some minor fixes of things like saving arrays in json that dont need saved

# RNN tutorial

Before you learn this, learn a normal network. How does an RNN work? RNN's work exactly like normal networks with a few key exceptions.
They allow you to build them with <b>RecEmitterLayer</b>'s and <b>RecReceiverLayer</b>'s.

## What do these special layers do?

In simple terms, a <b>RecEmitterLayer</b> saves its input and then forwards it to its corressponding <b>RecReceiverLayer</b> the next time you call <b>.forward()</b> on the model. It also forwards the data straight through it to the next layer.

<img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/rnn_diagram.png?raw=true">

## How to use these layers?

To add these layers into your model, add them like a normal layer.

```

let net = new Ment.Rnn([
new Input(2),
new FC(2,5),
new Sig(5),
new RecReceiverLayer('id'), //special layers here!!
new FC(10,10),
new Sig(10),
new FC(10,5),
new Sig(5),
new RecEmitterLayer('id'), //special layers here!!
new FC(5,1)
]);

```

This will make a network that looks like this. (white arrows are data flow)

<img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/net_diagram.png?raw=true">

We will be using this network to be trained to become a XOR gate.

```

for(var i = 0;i<50000;i++){
net.forward([0,1]);
net.forward([1,0]);
net.forward([1,1]);
net.forward([0,0]);

net.backward([0]); //output zero for input [0,0]
net.backward([0]); //output zero for input [1,1]
net.backward([1]); //output one for input [1,0]
net.backward([1]); //output one for input [0,1]
}

console.log(net.forward([0,1]));
console.log(net.forward([1,0]));
console.log(net.forward([1,1]));
console.log(net.forward([0,0]));

```

Notice the difference with normal training? This is because with rnn's you must forward batches of data in succession and then backward the corressponding outputs
afterwards in the opposite order.

This is shown below:

```

//normal training:
net.forward(input1)
net.backward(output1)
net.forward(input2)
net.backward(output2)

//rnn training
net.forward(input1)
net.forward(input2)

net.backward(output2)
net.backward(output1)

```

What this allows is rnn networks to have a 'hidden state' or 'memory'. We can show this by training a network to count numbers.
This network will need to remember what it outputted last time to work. Now we could just input this into the network but the point here
is to show that rnn networks can remember, so we will be blocking all input with a <b>block</b> layer. (this layers ignores input and outputs zeroes)

```

let net = new Ment.Rnn([
new Block(1),
new RecReceiverLayer(':-)'), //special layers here!!
new FC(2,1),
new RecEmitterLayer(':-)'), //special layers here!!
new Output(1),
]);

for(var i = 0;i<5000;i++){

net.forward([0]); //the input doesnt matter here, its blocked out

net.forward([0]); //the input doesnt matter here, its blocked out

net.forward([0]); //the input doesnt matter here, its blocked out

net.backward([3]);
net.backward([2]);
net.backward([1]);

net.resetRecurrentData(); // this function resets its memory
//if we don't call it it might keep counting forever but we just want 1 2 and 3
}

console.log(net.forward([Math.random()])); //the input doesnt matter because the model cant see it,
console.log(net.forward([Math.random()]));
console.log(net.forward([Math.random()]));

```

> [ 1.0009040832519531]

> [ 2.000751256942749 ]

> [ 2.999542713165283 ]

It learned to count from 1 to 3! (or forever if u keep running it).

## Clearing a models "memory"

A model stores data from its last forward propagation. You can clear this data with this function:

```

net.resetRecurrentData(); // this function resets its memory

```

It will replace the data with zeroes. In the last model of the tutorial we cleared it after every batch of data so the model
learned that when its memory is zero it starts at 1. If your training an rnn make sure to clear its memory after batches or whenever you need it cleared.

contact me: trevorblythe82@gmail.com

Contribute?

```

```
