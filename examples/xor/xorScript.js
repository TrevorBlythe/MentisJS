Ment.polluteGlobal();
let out = function (text) {
	document.getElementById("out").innerHTML += "<br>" + text;
};

let clear = function () {
	document.getElementById("out").innerHTML = "";
};

let ctx = document.getElementById("canvas").getContext("2d");

var net = new Net([new FC(2, 3, true), new Sig(), new FC(3, 1, true), new Sig()], "Momentum");

net.batchSize = 4;
net.learningRate = 0.1;

let inputs = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];

let outputs = [[0], [1], [1], [0]];

let doit = function () {
	clear();
	for (var i = 0; i < 4; i++) {
		net.train(inputs[i % 4], outputs[i % 4]);
	}
	let totalCost = 0;
	let r = Math.random();
	totalCost += getLoss(net.forward(inputs[0]), outputs[0]);
	if (r < 0.25) {
		render({ net: net, ctx: ctx, x: 10, y: 10, scale: 10, spread: 50 });
	}
	r -= 0.25;
	totalCost += getLoss(net.forward(inputs[1]), outputs[1]);
	if (r < 0.25) {
		render({ net: net, ctx: ctx, x: 10, y: 10, scale: 10, spread: 50 });
	}
	r -= 0.25;

	totalCost += getLoss(net.forward(inputs[2]), outputs[2]);
	if (r > 0.25) {
		render({ net: net, ctx: ctx, x: 10, y: 10, scale: 10, spread: 50 });
	}
	r -= 0.25;

	totalCost += getLoss(net.forward(inputs[3]), outputs[3]);
	if (r > 0.25) {
		render({ net: net, ctx: ctx, x: 10, y: 10, scale: 10, spread: 50 });
	}
	r -= 0.25;

	totalCost = totalCost / 4;
	if (totalCost < 0.001) {
		out("Total error of network: basically zero");
	} else {
		out("Total error of network: " + totalCost);
	}
	out("input: " + inputs[0] + ", output: " + net.forward([0, 0]).map((x) => x.toPrecision(5)));
	out("input: " + inputs[1] + ", output: " + net.forward([0, 1]).map((x) => x.toPrecision(5)));
	out("input: " + inputs[2] + ", output: " + net.forward([1, 0]).map((x) => x.toPrecision(5)));
	out("input: " + inputs[3] + ", output: " + net.forward([1, 1]).map((x) => x.toPrecision(5)));
};

let loop = setInterval(doit, 5);
// doit();
