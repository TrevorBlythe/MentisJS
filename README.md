# Mentis JS machine learning/evolution <img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/logo.png?raw=true" width = 100px>

A javascript neural network library. It can train with backpropagation or neural evoloution! Ill post some examples of those soon!
I am about to be a college student so expect the next four years to be this projects prime. This project is aimed to be the most simple
and intuitive machine learning library out there. Inspired by the complicated python ML apis (to make a better one )! Also Inspired by convnetjs!

Im writing it to be as simple as possible. It is now extremely easy to write your own layers! Look in FCLayer.js for a specification on how to write
your first one.

To use this library just copy the file "Mentis.js" which contains everything.

# Newest Update

<h3>ADDED MOMENTUM AND GROUND WORK FOR FUTURE OPTIMIZERS OTHER THAN SGD
and also..</h3>

0. Lots of tiny optimizations all around the library.
1. the "backwards" function in layer objects now takes error as a parameter instead of expected values
2. layer objects dont calculuate loss, it gets done by the network object.
3. Added a clearGrad() method in case you need to clear gradients. (doesnt reset iteration counter)
4. made saving all the activations in an rnn to a function "appendCurrentState" explanation in the function.
5. Got rid of weird error messages
6. Got rid of big error that made conv layers train disproportionately slow
7. Upscaling layers didnt work, now they do.
8. Deconv layers average out their gradients. (trains better)
9. A couple errors where i forgot to index an object or used the wrong counter.
10. Added a new Layer to insert Input whereever you want in a network.
11. Deleted useless files

Making the backward function in Layers take "error" as a paremeter instead of "expected output," is the biggest change and could be breaking change.
It needed to be done though as it optimizes things a lot and lets a lot of code be cleared out. If you have the old library update it because it
is much better now.

Keep in mind that this only changed LAYERS and not the actual net object, so old code should work fine still as long as you didnt train layers
individualy. Backward function in Net objects still take "expected output" as the parameter.

# Tutorial

<h2>How to make a Neural Network</h2>

<p>In this example we will make a neural network that will act as a XOR gate. (a logic gate)</p>
<p>First lets define the Dataset.</p>

```javascript
let inputs = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];

let outputs = [[0], [1], [1], [0]];
```

<p>Notice how the inputs and outputs correspond to the I/O of a XOR logic gate?</p>
<p>Next we must define the Network to be the gate. You make a network by creating</p>
<p>a "Net" object. Its constructor takes a list of layers that it will have</p>

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

<p>Notice how the first layer takes an input size of two? This is because we will be inputting two numbers into the network</p>
<p>The last layer only has one neuron. That will be the output neuron and will tell us what the network predicts.</p>
<p>This network is overkill for a simple XOR gate but is good for a tutorial. Next we put these layers in a Net object</p>

<p> Now that we have our network made we can use it to solve XOR! You can input data with the forward function.</p>

```javascript
network.forward(inputs[0]); //returns the output (a list with 1 number)
```

<p>Since our network isnt trained yet, it returns gibberish... Lets train it</p>

```javascript
for (var i = 0; i < 5000; i++) {
	net.train(inputs[i % 4], outputs[i % 4]);
}

//net.train() takes input and desired output as parameters.
```

<p>Now that it is trained lets test it</p>

```javascript
network.forward([0, 0]);
```

> [0.0001]

```javascript
network.forward([1, 0]);
```

> [0.9983]

```javascript
network.forward([0, 1]);
```

> [0.9923]

```javascript
network.forward([1, 1]);
```

> [0.0012]

It works!

# Using the optimizers

I recently added an optimizer that can increase training speed greatly. I should have done it earlier, it was super simple but made things train a lot better.

Here is how to use different optimizers.

### 1. Just include it in the Net() constructor.

```javascript
/*
You can either use the string of the name or make the object in the parameters

String method sucks but had to have it for backwards compatibility.

With the "make the object in parameter" method you can include hyper-parameters, in this case you can set the momentum constant (default is 0.9).

*/
let network = new Ment.Net([layers]); //uses SGD optimizer (defualt)
let network = new Ment.Net([layers], "SGD"); // also uses SGD
let network = new Ment.Net([layers], "Momentum"); // uses momentum
let network = new Ment.Net([layers], new Ment.Momentum(0.5)); //uses momentum with 0.5 momentum constant
```

If you wanna try out momentum go to the mnist example and press "reload network." It trains way faster.

### How to change optimizers mid-train:

```javascript
//Assume net has SGD and we wanna switch to momentum
net.optimizer = new Momentum(0.9);
net.optimizer.netObject = net;
net.optimizer.initialize();
//thats it
```

# RNN's

Before you learn this, learn a normal network. How does an RNN work? RNN's work almost exactly like normal networks with a few key exceptions.
They allow you to build them with <b>RecEmitterLayer</b>'s and <b>RecReceiverLayer</b>'s.

## What do these special layers do?

In simple terms, a <b>RecEmitterLayer</b> saves its input and then forwards it to its corressponding <b>RecReceiverLayer</b> the next time you call <b>.forward()</b> on the model. It also forwards the data straight through it to the next layer.

<img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/rnn_diagram.png?raw=true">

## How to use these layers?

To add these layers into your model, instanstiate an RNN object and add them like a normal layer.

```javascript
let net = new Ment.Rnn([
	new Ment.Input(2),
	new Ment.FC(2, 5),
	new Ment.Sig(5),
	new Ment.RecReceiverLayer("id"), //special layers here!!
	new Ment.FC(10, 10),
	new Ment.Sig(10),
	new Ment.FC(10, 5),
	new Ment.Sig(5),
	new Ment.RecEmitterLayer("id"), //special layers here!!
	new Ment.FC(5, 1),
]);
```

Notice the parameter of the layers? That is an identification string. That is how you link corressponding <b>RecReceiver</b> and <b>RecEmitter</b> layers. It can be anything you want as long as its the same for both.
This will make a network that looks like this. (white arrows are data flow)

<img src="https://github.com/TrevorBlythe/MentisJS/blob/main/images/net_diagram.png?raw=true" width = "800px">

We will be using this network to be trained to become a XOR gate. (im open to better example ideas)

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

console.log(net.forward([0, 1]));
console.log(net.forward([1, 0]));
console.log(net.forward([1, 1]));
console.log(net.forward([0, 0]));
```

Notice the difference with normal training? This is because with rnn's you must forward batches of data in succession and then backward the corressponding outputs
afterwards in the opposite order.

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

What this allows is rnn networks to have a 'hidden state' or 'memory'. We can show this by training a network to count numbers.
This network will need to remember what it outputted last time to work. Now we could just input this into the network but the point here
is to show that rnn networks can remember, so we will be blocking all input with a <b>block</b> layer. (this layers ignores input and outputs zeroes)

```javascript
let net = new Ment.Rnn([
	new Ment.Block(1),
	new Ment.RecReceiverLayer(":-)"), //special layers here!!
	new Ment.FC(2, 1),
	new Ment.RecEmitterLayer(":-)"), //special layers here!!
	new Ment.Output(1),
]);

for (var i = 0; i < 5000; i++) {
	net.forward([0]); //the input doesnt matter here, its blocked out ( by the block layer)

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

```javascript
net.resetRecurrentData(); // this function resets its memory
```

It will replace the memory with zeroes. In the last model of the tutorial we cleared it after every batch of data so the model
learned that when its memory is zero it starts at 1. If your training an rnn make sure to clear its memory after batches or whenever you need it cleared.
Remember, Rnn's take up more memory the more times you call forwards. It saves its activations for training. The memory is cleared when you call the backwards
function as many times as you called the forwards function. The states are saved in a list called "savedStates", (rnnObject.savedStates)
contact me: trevorblythe82@gmail.com

Open up index.html for lots of examples of this network!
