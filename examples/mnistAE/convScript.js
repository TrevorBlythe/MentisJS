Ment.polluteGlobal();
let examplesSeen = 0;

let ctx = document.getElementById("canvas").getContext("2d");
let ctxO = document.getElementById("canvasOut").getContext("2d");

let drawFromArr = function (arr, ctxy) {
	let width = 28;
	let height = 28;
	for (var i = 0; i < height; i++) {
		for (var j = 0; j < width; j++) {
			//array pixel values are stored as r,r,r,g,g,g,b,b,b
			r = arr[i * width + j] * 255;
			ctxy.fillStyle = "rgb(" + r + "," + r + "," + r + ")";

			ctxy.fillRect(j * 5, i * 5, 5, 5);
		}
	}
};

let getDigit = function (digit) {
	let c = Math.floor(Math.random() * 49);
	return digits[names[digit]].slice(c * 28 * 28, 28 * 28 + c * 28 * 28);
};

let greenBox = function (p) {
	return (
		"<span style='display:inline-block;height:20px;width:20px;background-color:rgb(0," + p + ",0);border:1px solid black'></span>"
	);
};
var net = new Ment.Net(
	[
		new Ment.DepthWiseConv([28, 28, 1], [10, 10], 6, 3),
		new Ment.DepthWiseConv([7, 7, 6], [3, 3], 12, 1),
		new Ment.Conv([5, 5, 12], [1, 1], 12, 1),
		new Ment.FC(5 * 5 * 12, 10),
		new Ment.Tanh(),
		new Ment.FC(10, 5 * 5 * 12),

		new Ment.DepthWiseDeconv([7, 7, 12], [3, 3], 12, 1),
		new Ment.DeConv([7, 7, 12], [1, 1], 12, 1),
		new Ment.DepthWiseDeconv([28, 28, 1], [10, 10], 12, 3),
		new Ment.LRelu(0.01),
	],
	new Ment.AdamW(100)
	// new Ment.SGD()
);

net.batchSize = 1;
net.learningRate = 0.001;

document.getElementById("lr").innerHTML = `learning rate: ${net.learningRate}`;
let names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
var start = new Date();
let loss = 0;
let newInput = new Float64Array(28 * 28 * 3);
var go = function () {
	window.loop = setInterval(function () {
		start = new Date();
		//get digit returns a black and white image of a digit
		let input = getDigit(Math.floor(Math.random() * 9));

		loss = net.train(input, input);

		let output = net.outData;
		examplesSeen++;
		//do some think that takes a while here

		var runTime = new Date() - start;
		document.getElementById("es").innerHTML = "examples seen:" + examplesSeen + "Script took:" + runTime + " Milliseconds to run";
	}, 0);
	window.loopTwo = setInterval(() => {
		render();
		drawFromArr(net.inData, ctx);
		drawFromArr(net.outData, ctxO);
		document.getElementById("err").innerHTML = `Total error: ${loss.toString().substring(0, 3)}`;
	}, 2000);
};

var render = function () {
	document
		.getElementById("canvasNet")
		.getContext("2d")
		.clearRect(0, 0, document.getElementById("canvasNet").width, document.getElementById("canvasNet").height);
	renderBox({
		net: net,
		ctx: document.getElementById("canvasNet").getContext("2d"),
		x: 5,
		y: 5,
		scale: 50,
		spread: 30,
		background: false,
		showAsImage: true,
	});
};
go();
render();

console.log("To all the haters: Your negativity only fuels my determination to succeed..");
