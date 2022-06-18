let out = function (text) {
	document.getElementById('out').innerHTML += '<br>' + text;
};

let clear = function () {
	document.getElementById('out').innerHTML = '';
};

let examplesSeen = 0;

let ctx = [
	document.getElementById('canvas0').getContext('2d'),
	document.getElementById('canvas1').getContext('2d'),
	document.getElementById('canvas2').getContext('2d'),
	document.getElementById('canvas3').getContext('2d'),
	document.getElementById('canvas4').getContext('2d'),
	document.getElementById('canvas5').getContext('2d'),
	document.getElementById('canvas6').getContext('2d'),
	document.getElementById('canvas7').getContext('2d'),
	document.getElementById('canvas8').getContext('2d'),
	document.getElementById('canvas9').getContext('2d'),
];

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
	return digits[digit].slice(c * 28 * 28, 28 * 28 + c * 28 * 28);
};

let greenBox = function (p) {
	return "<span style='display:inline-block;height:20px;width:20px;background-color:rgb(0," + p + ",0);border:1px solid black'></span>";
};
//inWidth, inHeight, inDepth, filterWidth, filterHeight, filters=3, stride=1, padding=0

// var net = new swag.Net([
// 	new swag.Conv(28, 28, 1, 4, 4, 3, 1, 0),
// 	new swag.MaxPool(25, 25, 3, 5, 5, 5),
// 	new swag.Conv(5, 5, 3, 2, 2, 3, 1, 0),
// 	new swag.MaxPool(4, 4, 3, 2, 2, 1),
// 	new swag.FC(27, 13),
// 	new swag.Sig(13),
// 	new swag.FC(13, 10),
// 	new swag.Relu(10),
// ]);

var net = new swag.Net([
	new swag.Conv(28, 28, 1, 10, 10, 5, 3, 0),
	new swag.MaxPool(7, 7, 5, 2, 2, 2),
	new swag.FC(45, 35),
	new swag.Sig(35),
	new swag.FC(35, 10),
	new swag.Relu(10),
]);

net.batchSize = 20;
net.learningRate = 0.1;

let names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

var go = function () {
	window.loop = setInterval(function () {
		for (var i = 0; i < 10; i++) {
			let c = Array(10).fill(0);
			c[i] = 5;
			let err = net.train(getDigit(names[i]), c);
			document.getElementById('err' + names[i]).innerHTML = 'Total Error: ' + parseInt(err * 100) + '%';
			document.getElementById(names[i] + 'out').innerHTML = '';
			let max = 0;
			let ind = 0;
			for (var j = 0; j < 10; j++) {
				if (net.layers[net.layers.length - 1].outData[j] > max) {
					// console.log(max);
					max = net.layers[net.layers.length - 1].outData[j];
					ind = j;
				}
				document.getElementById(names[i] + 'out').innerHTML += greenBox(net.layers[net.layers.length - 1].outData[j] * 255);
			}
			if (ind == i) {
				document.getElementById(names[i] + 'c').innerHTML = 'Correct';
			} else {
				document.getElementById(names[i] + 'c').innerHTML = 'WRONG YOU IDIOT!';
			}
			drawFromArr(getDigit(names[i]), ctx[i]);
		}
		//draw the filters
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
	}, 10);
};

go();
