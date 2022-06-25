Ment.polluteGlobal();
let examplesSeen = 0;

let ctx = document.getElementById('canvas').getContext('2d');
let ctxO = document.getElementById('canvasOut').getContext('2d');

let ctxF = document.getElementById('filterCanvas').getContext('2d');
let ctxFG = document.getElementById('filterGCanvas').getContext('2d');

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

let drawFromArrF = function (arr, ctx, xx, y, outOf) {
	let width = net.layers[0].filterWidth;
	let height = net.layers[0].filterHeight;
	xx = xx || 0;
	y = y || 0;
	for (var i = 0; i < height; i++) {
		for (var j = 0; j < width; j++) {
			let x = arr[i * width + j] * outOf;

			if (x < 0) {
				ctx.fillStyle = 'rgb(' + -x + ',' + 0 + ',' + 0 + ')';
			} else {
				ctx.fillStyle = 'rgb(' + 0 + ',' + x + ',' + 0 + ')';
			}
			ctx.fillRect(xx + j * 5, y + i * 5, 5, 5);
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

let one = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
var net = new Ment.Net([
	new Ment.Conv(28, 28, 1, 10, 10, 5, 3, 0),
	new Ment.Conv(7, 7, 5, 3, 3, 3, 1, 0),
	new Ment.DeConv(7, 7, 5, 3, 3, 3, 1, 0),
	new Ment.DeConv(28, 28, 1, 10, 10, 5, 3, 0),
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
		drawFromArr(input, ctx);
		drawFromArr(output, ctxO);
		document.getElementById('es').innerHTML = 'examples seen:' + examplesSeen;
		examplesSeen++;
	}, 0);
};

go();
