# swagNeuralNet
A javascript neural network library. It can train with backpropagation or neural evoloution! Ill post some examples of those soon!


# This project is unfinished!
I probably will leave it here. 

Just open index.html and i have A XOR example ready training for you. Can someone finish this for me?

Things i might add:
lstm layer
conv layer
things like adadelta and momentum 

# Tutorial:

How to make a network.

var net = swag.Net(layers,minmax);

minmax is just how big the weights are initialized as. you can leave it blank.

example:

var net = new swag.Net([
	new swag.Layer('fc', 2),
	new swag.Layer('fc', 10, 'sigmoid'),
	new swag.Layer('fc', 1, 'sigmoid'),
]);


forwarding:

net.forward([0,0]); //outputs a copy of the activations in the last layer.... its a copy so its safe to change the contents.

saving/loading:
net.save(); //returns json string of model
net.save(true); //prompts user to save a file of the json
net.load(json); turns "net" into whatever the json is
netTwo.copy(net); // you can also use this to make copies of neural networks. Its safe.

training:

net.train(input , expected output);

net.evolve(dataset, iterations);
More examples coming soon.


Population training:

this is a method of training where we make a bunch of models and test which one is the best. I made it easy for you

let pop = new swag.Population(popSize, elitism, mutationRate, netLayers);


//* lets say the third network performed the best.... we set its score to be high

pop.networks[2].score = 9999;

//* now we mutate the population and breed the best ones! The VERY best networks just up and go on to the next pop without mutations

pop.cullAndBreed();

//* now your ready to score the networks again! all scores got automatically set to Zero!





