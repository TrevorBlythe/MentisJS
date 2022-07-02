Ment.polluteGlobal();
let examplesSeen = 0;

let ctx = document.getElementById('canvas').getContext('2d');
let ctxO = document.getElementById('canvasOut').getContext('2d');

let drawFromArr = function (arr, ctxy) {
	let width = 28;
	let height = 28;
	for (var i = 0; i < height; i++) {
		for (var j = 0; j < width; j++) {
			let x = arr[i * width + j] * 255;
			ctxy.fillStyle = 'rgb(' + x + ',' + x + ',' + x + ')';
			ctxy.fillRect(j * 5, i * 5, 5, 5);
		}
	}
};

let getDigit = function (digit) {
	let c = Math.floor(Math.random() * 49);
	return digits[names[digit]].slice(c * 28 * 28, 28 * 28 + c * 28 * 28);
};

let greenBox = function (p) {
	return "<span style='display:inline-block;height:20px;width:20px;background-color:rgb(0," + p + ",0);border:1px solid black'></span>";
};

var net = new Ment.Net([
	new Ment.Conv([28, 28, 1], [10, 10], 5, 3),
	new Ment.Conv([7, 7, 5], [3, 3], 3, 1),
	new Ment.DeConv([7, 7, 5], [3, 3], 3, 1),
	new Ment.DeConv([28, 28, 1], [10, 10], 5, 3),
]);

net.batchSize = 10;
net.learningRate = 0.01;

document.getElementById('lr').innerHTML = `learning rate: ${net.learningRate}`;
let names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

var go = function () {
	window.loop = setInterval(function () {
		let input = getDigit(Math.floor(Math.random() * 9));
		let loss = net.train(input, input);

		document.getElementById('err').innerHTML = `Total error: ${loss.toString().substring(0, 3)}`;
		let output = net.outData;
		if (net.epoch % 200 == 0 || net.epoch == 1) {
			drawFromArr(input, ctx);
			drawFromArr(output, ctxO);
		}
		document.getElementById('es').innerHTML = 'examples seen:' + examplesSeen;
		examplesSeen++;
	}, 0);
};

var render = function () {
	document
		.getElementById('canvasNet')
		.getContext('2d')
		.clearRect(0, 0, document.getElementById('canvasNet').width, document.getElementById('canvasNet').height);
	renderBox(net, document.getElementById('canvasNet').getContext('2d'), 5, 5, 60, 2, 'black');
};
go();
render();
