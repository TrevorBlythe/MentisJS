let out = function (text) {
	document.getElementById('out').innerHTML += '<br>' + text;
};

let clear = function () {
	document.getElementById('out').innerHTML = '';
};

let ctxZ = document.getElementById('canvas0').getContext('2d');
let ctxO = document.getElementById('canvas1').getContext('2d');
let ctxT = document.getElementById('canvas2').getContext('2d');
let ctxTh = document.getElementById('canvas3').getContext('2d');

let drawFromArr = function (arr, ctx) {
	let width = 28;
	let height = 28;
	for (var i = 0; i < height; i++) {
		for (var j = 0; j < width; j++) {
			let x = arr[i * width + j] * 255;
			ctx.fillStyle = 'rgb(' + x + ',' + x + ',' + x + ')';
			ctx.fillRect(j * 5, i * 5, 5, 5);
		}
	}
};

let getDigit = function (digit) {
	let c = Math.floor(Math.random() * 100);
	return digits[digit].slice(c * 28 * 28, 28 * 28 + c * 28 * 28);
};

let greenBox = function (p) {
	return (
		"<span style='margin-right:3%;display:inline-block;height:20px;width:20px;background-color:rgb(0," +
		p +
		",0);border:1px solid black'></span>"
	);
};
//inWidth, inHeight, inDepth, filterWidth, filterHeight, filters=3, stride=1, padding=0
var net = new swag.Net([
	new swag.Conv(28, 28, 1, 4, 4, 3, 1, 0),
	new swag.MaxPool(25, 25, 3, 5, 5, 5),
	new swag.FC(75, 30),
	new swag.Sig(30),
	new swag.FC(30, 4),
	new swag.Sig(4),
]);

net.batchSize = 5;
net.learningRate = 0.1;

// let counter = 0;
// setInterval(function(){
// 	counter++;
// 	drawFromArr(digits.three.slice(counter * 28 * 28,28 * 28 + counter * 28 * 28),28,28);
// },100);

setInterval(function () {
	for (var i = 0; i < 10; i++) {
		let c = getDigit('three');
		net.train(c, [0, 0, 0, 1]);
		c = getDigit('two');
		net.train(c, [0, 0, 1, 0]);
		c = getDigit('one');
		net.train(c, [0, 1, 0, 0]);
		c = getDigit('zero');
		net.train(c, [1, 0, 0, 0]);
	}
	let c = getDigit('three');
	let err = net.train(c, [0, 0, 0, 1]);
	document.getElementById('errTh').innerHTML =
		'Total Error: ' + parseInt(err * 100) + '%';
	document.getElementById('threeout').innerHTML =
		'zero: ' +
		greenBox(net.layers[net.layers.length - 1].outData[0] * 255) +
		' one: ' +
		greenBox(net.layers[net.layers.length - 1].outData[1] * 255) +
		' two: ' +
		greenBox(net.layers[net.layers.length - 1].outData[2] * 255) +
		' three: ' +
		greenBox(net.layers[net.layers.length - 1].outData[3] * 255) +
		' ';
	drawFromArr(c, ctxTh);

	c = getDigit('two');
	err = net.train(c, [0, 0, 1, 0]);
	document.getElementById('errT').innerHTML =
		'Total Error: ' + parseInt(err * 100) + '%';

	document.getElementById('twoout').innerHTML =
		'zero: ' +
		greenBox(net.layers[net.layers.length - 1].outData[0] * 255) +
		' one: ' +
		greenBox(net.layers[net.layers.length - 1].outData[1] * 255) +
		' two: ' +
		greenBox(net.layers[net.layers.length - 1].outData[2] * 255) +
		' three: ' +
		greenBox(net.layers[net.layers.length - 1].outData[3] * 255) +
		' ';
	drawFromArr(c, ctxT);

	c = getDigit('one');
	err = net.train(c, [0, 1, 0, 0]);
	document.getElementById('errO').innerHTML =
		'Total Error: ' + parseInt(err * 100) + '%';

	document.getElementById('oneout').innerHTML =
		'zero: ' +
		greenBox(net.layers[net.layers.length - 1].outData[0] * 255) +
		' one: ' +
		greenBox(net.layers[net.layers.length - 1].outData[1] * 255) +
		' two: ' +
		greenBox(net.layers[net.layers.length - 1].outData[2] * 255) +
		' three: ' +
		greenBox(net.layers[net.layers.length - 1].outData[3] * 255) +
		' ';
	drawFromArr(c, ctxO);

	c = getDigit('zero');
	err = net.train(c, [1, 0, 0, 0]);
	document.getElementById('errZ').innerHTML =
		'Total Error: ' + parseInt(err * 100) + '%';

	document.getElementById('zeroout').innerHTML =
		'zero: ' +
		greenBox(net.layers[net.layers.length - 1].outData[0] * 255) +
		' one: ' +
		greenBox(net.layers[net.layers.length - 1].outData[1] * 255) +
		' two: ' +
		greenBox(net.layers[net.layers.length - 1].outData[2] * 255) +
		' three: ' +
		greenBox(net.layers[net.layers.length - 1].outData[3] * 255) +
		' ';
	drawFromArr(c, ctxZ);
}, 100);
