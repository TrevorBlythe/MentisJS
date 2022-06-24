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
//inWidth, inHeight, inDepth, filterWidth, filterHeight, filters=3, stride=1, padding=0

var net = new Ment.Net([
	new Ment.Conv(28, 28, 1, 10, 10, 5, 3, 0),
	new Ment.MaxPool(7, 7, 5, 2, 2, 2),
	new Ment.FC(45, 35),
	new Ment.Tanh(),
	new Ment.FC(35, 5, false),
	new Ment.FC(5, 35, false),
	new Ment.Relu(),
	new Ment.FC(35, 100, false),
	new Ment.Relu(),
	new Ment.FC(100, 28 * 28, false),
	new Ment.Relu(28 * 28),
]);

net.batchSize = 10;
net.learningRate = 0.01;

let names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

var go = function () {
	window.loop = setInterval(function () {
		let input = getDigit(Math.floor(Math.random() * 10));
		let loss = net.train(input, input);
		document.getElementById('err').innerHTML = `Total error: ${loss.toString().substring(0, 3)}`;
		let output = net.outData;
		drawFromArr(input, ctx);
		drawFromArr(output, ctxO);
		for (var i = 0; i < net.layers[0].filters; i++) {
			ctxF.clearRect(i * net.layers[0].filterWidth * 20, 0, 100, 100);
			drawFromArrF(
				net.layers[0].filterw.slice(
					i * net.layers[0].filterWidth * net.layers[0].filterHeight,
					i * net.layers[0].filterWidth * net.layers[0].filterHeight + net.layers[0].filterWidth + net.layers[0].filterWidth * (net.layers[0].filterHeight - 1)
				),
				ctxF,
				5 + i * net.layers[0].filterWidth * 6,
				5,
				255
			);
		}

		for (var i = 0; i < net.layers[0].filters; i++) {
			ctxFG.clearRect(i * net.layers[0].filterWidth * 20, 0, 100, 100);
			drawFromArrF(
				net.layers[0].filterws.slice(
					i * net.layers[0].filterWidth * net.layers[0].filterHeight,
					i * net.layers[0].filterWidth * net.layers[0].filterHeight + net.layers[0].filterWidth + net.layers[0].filterWidth * (net.layers[0].filterHeight - 1)
				),
				ctxFG,
				5 + i * net.layers[0].filterWidth * 6,
				5,
				2550
			);
		}
		document.getElementById('es').innerHTML = 'examples seen:' + examplesSeen;
		examplesSeen++;
	}, 0);
};

go();
