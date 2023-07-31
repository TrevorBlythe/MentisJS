var net;
let drawFromArr = function (arr, ctxy, xx, yy, width, height) {
	// let width = 28;
	// let height = 28;
	xx = xx || 0;
	y = yy || 0;
	for (var i = 0; i < height; i++) {
		for (var j = 0; j < width; j++) {
			let x = arr[i * width + j] * 255;
			ctxy.fillStyle = "rgb(" + x + "," + x + "," + x + ")";
			ctxy.fillRect(j * 5 + xx, i * 5 + yy, 5, 5);
		}
	}
};

let drawRG = function (arr, ctx, xx, y, outOf) {
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

var analyze = function () {
	let output = document.getElementById("networkTerminal");
	output.innerHTML = "";
	for (i in net.layers) {
		let layer = net.layers[i];
		let wrapper = output.appendChild(document.createElement("div"));
		wrapper.classList.add("layer");
		wrapper.appendChild(document.createElement("h2")).innerHTML = `<h2>Layer ${i} (${layer.constructor.name})</h2>`;
		let cols = wrapper.appendChild(document.createElement("div"));
		cols.classList.add("gjs-row");
		let leftCol = cols.appendChild(document.createElement("div"));
		let rightCol = cols.appendChild(document.createElement("div"));
		leftCol.classList.add("gjs-cell");
		leftCol.classList.add("leftCol");
		rightCol.classList.add("rightCol");
		rightCol.classList.add("gjs-cell");

		Object.keys(layer).forEach(function (key, index) {
			var value = layer[key];
			// console.log(key, index, value);
			if (value && !value.length && typeof value != "object") {
				leftCol.appendChild(document.createElement("p")).innerHTML = `${key}:${value}`;
			}
		});
		leftCol.appendChild(document.createElement("p")).innerHTML = `outsize: ${
			layer.outSizeDimensions ? layer.outSizeDimensions() : layer.outSize()
		}`;

		displayWeights(layer, rightCol);
	}
	render();
};

var displayWeights = function (layer, elem) {
	//drawFromArr arr, ctxy, xx, yy, width, height
	if (layer.filterw) {
		elem.appendChild(document.createElement("h2")).innerHTML = "Weights:";
		let tabContainer = elem.appendChild(document.createElement("div"));
		tabContainer.classList.add("tab-container");
		for (var i = 0; i < layer.filters; i++) {
			for (var d = 0; d < layer.inDepth; d++) {
				let tab = tabContainer.appendChild(document.createElement("div"));
				tab.classList.add("tab");
				tab.appendChild(document.createElement("p")).innerHTML = `Filter ${i}`;
				// tab.innerHTML = "test";
				let canvas = tab.appendChild(document.createElement("canvas"));
				let ctx = canvas.getContext("2d");
				canvas.width = layer.filterWidth;
				canvas.height = layer.filterHeight;
				canvas.style.width = `${layer.filterWidth / 2}vw`;
				canvas.style.height = `${layer.filterHeight / 2}vw`;

				for (var x = 0; x < layer.filterWidth; x++) {
					for (var y = 0; y < layer.filterHeight; y++) {
						let a =
							layer.filterw[
								i * layer.inDepth * layer.filterWidth * layer.filterHeight +
									x +
									y * layer.filterWidth +
									d * layer.filterWidth * layer.filterHeight
							] * 255;
						ctx.fillStyle = "rgb(" + (a < 0 ? -a : 0) + "," + a + ",0)";
						ctx.fillRect(x, y, 1, 1);
					}
				}
			}
		}
		elem.appendChild(document.createElement("h2")).innerHTML = "Biases:";
		tabContainer = elem.appendChild(document.createElement("div"));
		tabContainer.classList.add("tab-container");
		for (var d = 0; d < layer.inDepth; d++) {
			let tab = tabContainer.appendChild(document.createElement("div"));
			tab.classList.add("tab");
			tab.appendChild(document.createElement("p")).innerHTML = `Biases:`;
			// tab.innerHTML = "test";
			let canvas = tab.appendChild(document.createElement("canvas"));
			let ctx = canvas.getContext("2d");
			canvas.width = layer.outSizeDimensions()[0];
			canvas.height = layer.outSizeDimensions()[1];
			canvas.style.width = `${layer.outSizeDimensions()[0] / 2}vw`;
			canvas.style.height = `${layer.outSizeDimensions()[1] / 2}vw`;

			for (var x = 0; x < layer.outSizeDimensions()[0]; x++) {
				for (var y = 0; y < layer.outSizeDimensions()[1]; y++) {
					let a =
						layer.b[x + y * layer.outSizeDimensions()[0] + d * layer.outSizeDimensions()[1] * layer.outSizeDimensions()[0]] * 255;
					ctx.fillStyle = "rgb(" + (a < 0 ? -a : 0) + "," + a + ",0)";
					ctx.fillRect(x, y, 1, 1);
				}
			}
		}
	}
	if (layer.w) {
		elem.appendChild(document.createElement("h2")).innerHTML = "Weights:";
		let tabContainer = elem.appendChild(document.createElement("div"));
		tabContainer.classList.add("tab-container");
		for (var d = 0; d < layer.w.length; d++) {
			let tab = tabContainer.appendChild(document.createElement("div"));
			tab.classList.add("tab2");
			// tab.innerHTML = "test";
			let canvas = tab.appendChild(document.createElement("canvas"));
			let ctx = canvas.getContext("2d");
			canvas.width = 1;
			canvas.height = 1;
			canvas.style.width = `${1}vw`;
			canvas.style.height = `${1}vw`;

			let a = layer.w[d] * 255;
			ctx.fillStyle = "rgb(" + (a < 0 ? -a : 0) + "," + a + ",0)";
			ctx.fillRect(0, 0, 1, 1);
		}
	}
};

var render = function () {
	document
		.getElementById("canvasNet")
		.getContext("2d")
		.clearRect(0, 0, document.getElementById("canvasNet").width, document.getElementById("canvasNet").height);
	Ment.renderBox({
		net: net,
		ctx: document.getElementById("canvasNet").getContext("2d"),
		x: 5,
		y: 5,
		scale: 60,
		spread: 60 / net.layers.length,
		background: false,
		// showAsImage: true,
	});
};

net = Ment.Net.load(model);
analyze();

document.getElementById("import").onclick = function () {
	console.log("button clicked");
	var files = document.getElementById("selectFiles").files;
	console.log(files);
	if (files.length <= 0) {
		return false;
	}

	var fr = new FileReader();

	fr.onload = function (e) {
		console.log(e);
		// var result = JSON.parse(e.target.result);
		net = Ment.Net.load(e.target.result);
		analyze();
		// var formatted = JSON.stringify(result, null, 2);
		// document.getElementById("result").value = formatted;
	};

	fr.readAsText(files[0]);
	// fr.readAsText(files.item(0));
};
