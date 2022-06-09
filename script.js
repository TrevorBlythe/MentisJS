
let out = function (text) {
	document.getElementById('out').innerHTML += '<br>' + text;
};

let clear = function () {
	document.getElementById('out').innerHTML = '';
};

let ctx = document.getElementById('canvas').getContext('2d');

var net = new swag.Net([
	new swag.FC(2,3),
	new swag.Sig(3),
	new swag.FC(3,1),
]);

net.batchSize = 4;
net.learningRate = 1;

let inputs = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];

let outputs = [[0], [1], [1], [0]];

let doit = function () {
	clear();
	swag.render(net, ctx, 10, 10, 20);
	for (var i = 0; i < 100; i++) {
		var loss = net.train(inputs[i % 4], outputs[i % 4]);
		if (i % 5000 === 0) {
			let totalCost = 0;
			totalCost += swag.getLoss(net.forward(inputs[0]), outputs[0]);
			totalCost += swag.getLoss(net.forward(inputs[1]), outputs[1]);
			totalCost += swag.getLoss(net.forward(inputs[2]), outputs[2]);
			totalCost += swag.getLoss(net.forward(inputs[3]), outputs[3]);
			totalCost = totalCost / 4;
			if(totalCost < 0.001){
				out("Total error of network: basically zero");
			}else{
				out("Total error of network: " + totalCost);
			}
		}
	}

	out(inputs[0] + ': ' + net.forward([0, 0]));
	out(inputs[1] + ': ' + net.forward([0, 1]));
	out(inputs[2] + ': ' + net.forward([1, 0]));
	out(inputs[3] + ': ' + net.forward([1, 1]));
};

setInterval(doit, 300);
// doit();
