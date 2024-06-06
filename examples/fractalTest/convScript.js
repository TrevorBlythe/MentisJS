let ctx = document.getElementById("canvas").getContext("2d");

let drawFromArr = function (arr, ctxy, width, height) {
	for (var i = 0; i < height; i++) {
		for (var j = 0; j < width; j++) {
			let x = arr[i * width + j];
			ctxy.fillStyle = "rgb(" + x + "," + x + "," + x + ")";
			ctxy.fillRect(j, i, 1, 1);
		}
	}
	// console.log(arr);
};

let drawFromArrColor = function (arr, ctxy, width, height) {
	//arr = [r,g,b,a,r,g,b,a,r,g,b,a,...]
	for (var i = 0; i < height; i++) {
		for (var j = 0; j < width; j++) {
			let x = i * width * 4 + j * 4;
			ctxy.fillStyle = "rgba(" + arr[x] + "," + arr[x + 1] + "," + arr[x + 2] + "," + arr[x + 3] + ")";
			ctxy.fillRect(j, i, 1, 1);
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
				ctx.fillStyle = "rgb(" + -x + "," + 0 + "," + 0 + ")";
			} else {
				ctx.fillStyle = "rgb(" + 0 + "," + x + "," + 0 + ")";
			}
			ctx.fillRect(xx + j * 5, y + i * 5, 5, 5);
		}
	}
};

var i = 0;
var go = function () {
	window.loop = setInterval(function () {
		drawFromArr(
			generateSimplifiedMandelbrot(
				100,
				100,
				i,
				(Math.random() - 0.5) * 5,
				(Math.random() - 0.5) * 5,
				Math.random() / i,
				Math.random() / i
			),
			ctx,
			100,
			100
		);
		i++;
		if (i > 100) {
			clearInterval(window.loop);
		}
	}, 100);
};
go();
