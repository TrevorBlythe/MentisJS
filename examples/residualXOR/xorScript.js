Ment.polluteGlobal();
let examplesSeen = 0;

var net = new Ment.Net([
	new FC(2, 3),
	new Sig(),
	new FC(3, 3),
	new Sig(),
	new FC(3, 3),
	new Sig(),
	new FC(3, 3),
	new Sig(),
	new FC(3, 3),
	new Sig(),
	new FC(3, 3),
	new Sig(),
	new FC(3, 3),
	new Sig(),
	new FC(3, 1),
]);

net.batchSize = 1;
net.learningRate = 0.3;

let inputs = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];

let outputs = [[0], [1], [1], [0]];

document.getElementById("lr").innerHTML = `learning rate: ${net.learningRate}`;

var go = function () {
	window.loop = setInterval(function () {
		let loss = 0;
		let outputText = "Network outputs: ";
		for (var i = 0; i < 100; i++) {
			loss += net.train(inputs[i % 4], outputs[i % 4]);
			if (i < 4) {
				outputText += Math.round(10 * net.outData[0]) / 10 + ", ";
			}
		}
		if (loss < 0.1) {
			loss = 0;
			document.getElementById("message").innerHTML = "Residual layers made it work! Congrats! 😊";
		}
		document.getElementById("err").innerHTML = outputText + ` <br><br>Total error: ${loss.toString().substring(0, 3)}`;
		document.getElementById("es").innerHTML = "examples seen:" + examplesSeen;
		examplesSeen++;
	}, 0);
	window.loopTwo = setInterval(() => {
		render();
	}, 1000);
};

var render = function () {
	document
		.getElementById("canvasNet")
		.getContext("2d")
		.clearRect(0, 0, document.getElementById("canvasNet").width, document.getElementById("canvasNet").height);
	Ment.renderBox({
		net: net,
		ctx: document.getElementById("canvasNet").getContext("2d"),
		x: 0,
		y: 0,
		scale: 50,
		spread: 20,
		background: false,
	});
};
go();
render();
