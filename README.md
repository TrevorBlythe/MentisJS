# Swagnet JS machine learning/evoloution
A javascript neural network library. It can train with backpropagation or neural evoloution! Ill post some examples of those soon!
I am about to be a college student so expect the next four years to be this projects prime. This project is aimed to be the most simple
and intuitive machine learning library out there. Inspired by the complicated python ML apis (to make a better one )! Also Inspired by convnetjs!

# Showcase

![image](https://github.com/TrevorBlythe/swagNeuralNet/blob/main/showcase/MnistDigits.png?raw=true)

![image](https://github.com/TrevorBlythe/swagNeuralNet/blob/main/showcase/XORrender.png?raw=true)

# This project is coming along!
Im rewriting it to be as simple as possible. It is now extremely easy to write your own layers! Look in FCLayer.js for a specification on how to write 
your first one.


Things i will add:
deconvoloution, soon
lstm layer, soon
padding in conv layer, at 30 github stars
more layers
optimizers like adadelta


Things for the distant future:

GAN tutorial.
Diffusion model tutorial.
WASM for speed?


# Tutorial:

How to make a network.

var net = swag.Net(array of layers);

var net = new swag.Net([
	new swag.FC(2,3),
	new swag.Sig(3),
	new swag.FC(3,1),
]);

var net = new swag.Net([
	new swag.Conv(28,28,1,4,4,3,1,0),
	new swag.MaxPool(25,25,3,5,5,5),
	new swag.FC(75,30),
	new swag.Sig(30),
	new swag.FC(30,10),
	new swag.Sig(10)
]);



# forwarding:

net.forward(array); this returns the output of a network with a given input ( array ) 

saving/loading:
net.save(); //broken, will fix 
net.save(true); //broken, will fix
net.load(json); // broken will fix soon
netTwo.copy(net); //broken, will fix soon

# training:
net.train(input , expected output);

net.evolve(dataset, iterations); //broken, will fix

# Population training:
Im rewriting this.

# License

Copyright (C) 2022 Trevor Blythe

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

https://www.gnu.org/licenses/quick-guide-gplv3.html

TLDR: Do whatever you want. You cant hold me liable for anything. You cant sublicense. 
If you make this library your own you gotta link to the original.
