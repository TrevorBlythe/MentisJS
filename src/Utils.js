var Ment = Ment || {};
{
	var return_v = false;
	var v_val = 0.0;

	var protectNaN = function (num) {
		if (num == Infinity) {
			return 1;
		} else if (num == -Infinity) {
			return -1;
		} else if (isNaN(num)) {
			return 0;
		} else {
			return num;
		}
	};

	var printNet = function (net) {
		let text = "LAYERS:\n";
		for (var i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];

			text += `	${net.layers[i].constructor.name} (${layer.inSizeDimensions ? "," + layer.inSizeDimensions() : layer.inSize()},${
				layer.outSizeDimensions ? "," + layer.outSizeDimensions() : layer.outSize()
			})${layer.id ? ` id:${layer.id}` : ""}\n`;
		}
		text += "STATS:\n";
		Object.keys(net).forEach(function (key, index) {
			if (key != "layers") {
				var value = net[key];
				text += `	${key}:${value}\n`;
			}
		});
		console.log(text);
	};

	var inputError = function (layer, arr) {
		let ret = `INPUT SIZE WRONG ON ${layer.constructor.name + (layer.id ? `(${layer.id})` : null)}: `;
		ret += `expected size (${layer.inSizeDimensions ? "," + layer.inSizeDimensions() : layer.inSize()}), got (${arr.length})`;
		return ret;
	};

	var gaussRandom = function () {
		if (return_v) {
			return_v = false;
			return v_val;
		}
		var u = 2 * Math.random() - 1;
		var v = 2 * Math.random() - 1;
		var r = u * u + v * v;
		if (r == 0 || r > 1) return gaussRandom();
		var c = Math.sqrt((-2 * Math.log(r)) / r);
		v_val = v * c; // cache this
		return_v = true;
		return u * c;
	};

	var isBrowser = () => !(typeof window === "undefined");
	//wrap everything in a namespace to not pollute global

	let clamp = function (num, min, max) {
		min = min | 0;
		max = max | 1;
		return Math.max(Math.min(num, max), min);
	};

	let lin = function (num) {
		return num;
	};

	let bounce = function (num, minmax) {
		minmax = minmax | 3;
		if (Math.abs(num) > minmax) {
			var t = (minmax * num) / Math.abs(num);
			num = t - (num - t);
			return bounce(num, minmax);
		}
		return num;
	};

	let getLoss = function (results, expected) {
		var loss = 0;
		for (var i = 0; i < results.length; i++) {
			loss += Math.pow(results[i] - expected[i], 2);
		}
		loss = loss / results.length;
		return loss;
	};

	let polluteGlobal = function () {
		for (const [key, value] of Object.entries(Ment)) {
			if (isBrowser()) {
				window[key] = value;
			} else {
				global[key] = value; //node
			}
		}
	};

	Ment.seed = 5;
	Ment.seededRandom = function (min, max) {
		max = max || 1;
		min = min || 0;

		Ment.seed = (Ment.seed * 9301 + 49297) % 233280;
		var rnd = Ment.seed / 233280;

		return min + rnd * (max - min);
	};

	let renderBox = function (options) {
		let net = options.net;
		let ctx = options.ctx;
		let x = options.x;
		let y = options.y;
		let scale = options.scale;
		let spread = options.spread;
		let background = options.background;
		if (background == undefined) {
			background = "white";
		}
		if (background == false) {
			background = "rgba(0,0,0,0)";
		}
		const SPREAD = spread || 3;
		let maxSize = 1;
		for (var i = 0; i < net.layers.length; i++) {
			if (net.layers[i].inSize() > maxSize) {
				maxSize = net.layers[i].inSize();
			} //end of if
			if (net.layers[i].outSize() > maxSize) {
				maxSize = net.layers[i].outSize();
			} //end of if
		}

		ctx.fillStyle = background;

		ctx.fillRect(x, y, net.layers.length * (spread + scale) + scale, 7 * scale);
		for (var i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];
			let layerSize = layer.inSize() > layer.outSize() ? layer.inSize() : layer.outSize();
			layerSize = layerSize / maxSize; //is zero to one. one if its the max size, 0 if its zero size
			let layerLeftSize = layer.inSize() / maxSize; // is zero to one, same as above
			let layerRightSize = layer.outSize() / maxSize; // same as above

			Ment.seed = layer.constructor.name.charCodeAt(0) * 5; //------ make seed from layer name
			for (var n = 1; n < layer.constructor.name.length; n++) {
				Ment.seed += layer.constructor.name.charCodeAt(n);
			} //----------End of make random seed from layer name
			ctx.fillStyle = `rgb(${Ment.seededRandom(150, 255)},${Ment.seededRandom(100, 255)},${Ment.seededRandom(100, 255)})`;

			let xy = scale / 2 + i * scale + i * spread + x; //x coord of top left, (scale)/2 is the margin on the left added.
			let yy = scale / 2 + y + (1 - layerLeftSize * 1) * scale * 3; // y coord of top left, scale/2 is the top margin.
			let xyy = scale / 2 + i * scale + i * spread + x + scale; //x coord of top right, each layer is "scale" pixels wide
			let yyy = scale / 2 + y + (1 - layerRightSize * 1) * scale * 3; // y coord of top right corner, not sure why the sqrt here works LOL

			ctx.beginPath();
			ctx.moveTo(xy, yy);
			ctx.lineTo(xyy, yyy);
			ctx.lineTo(xyy, yyy + layerRightSize * scale * 6);
			ctx.lineTo(xy, yy + layerLeftSize * scale * 6);
			ctx.lineTo(xy, yy);
			ctx.fill();

			layerSize = Math.max(0.5, layerSize);
			ctx.font = `${(layerSize * scale) / 2}px Verdana Geneva sans-serif`;
			ctx.fillStyle = "black";
			ctx.save();
			ctx.textAlign = "center";
			ctx.translate(xy + (xyy - xy) / 2, yyy + layerRightSize * scale * 3);
			ctx.rotate(-Math.PI / 2);
			ctx.fillText(layer.constructor.name, 0, 0);
			ctx.font = `${(layerSize * scale) / 3}px serif`;
			ctx.fillText(
				`(${layer.inSizeDimensions ? layer.inSizeDimensions() : layer.inSize()})(${
					layer.outSizeDimensions ? layer.outSizeDimensions() : layer.outSize()
				})`,
				0,
				scale / 3
			);
			ctx.restore();
			if (layer.outSizeDimensions && options.showAsImage) {
				wid = layer.outSizeDimensions()[0];
				hei = layer.outSizeDimensions()[1] | 1;
				dep = layer.outSizeDimensions()[2] | 1;
				for (var g = 0; g < dep; g++) {
					for (var h = 0; h < hei; h++) {
						for (var j = 0; j < wid; j++) {
							let row = scale * 6 + y + (scale / 25) * h;
							c = layer.outData[h * wid + j + g * (wid * hei)] * 255;
							ctx.fillStyle = "rgb(" + c + "," + c + "," + c + ")";
							ctx.fillRect(
								xy + -((hei * scale) / 25 - scale) / 2 + (scale / 25) * j,
								row + ((scale * hei) / 25) * g,
								scale / 25,
								scale / 25
							);
						}
					}
				}
			}
		}
	};

	let render = function (options) {
		let net = options.net;
		let ctx = options.ctx;
		let x = options.x;
		let y = options.y;
		let scale = options.scale;
		let spread = options.spread;
		let background = options.background;
		// a built in network renderer
		if (background == undefined) {
			background = true;
		}
		const SPREAD = spread || 10;
		const DRAWBACKGROUND = background;
		let maxSize = 1;
		for (var i = 0; i < net.layers.length; i++) {
			if (net.layers[i].constructor.name != "ConvLayer" && net.layers[i].constructor.name != "MaxPoolLayer") {
				if (net.layers[i].inSize() > maxSize) {
					maxSize = net.layers[i].inSize();
				} //end of if
				if (net.layers[i].outSize() > maxSize) {
					maxSize = net.layers[i].outSize();
				} //end of if
			} //end of for
			if (DRAWBACKGROUND) {
				ctx.fillStyle = "grey";
				ctx.fillRect(x, y, 5 + net.layers.length * scale * SPREAD + scale, 5 + maxSize * scale * 2);
			}
		}

		//render the neurons in each layer...
		for (i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];
			if (layer.constructor.name != "ConvLayer" && layer.constructor.name != "MaxPoolLayer") {
				for (var h = 0; h < layer.outSize(); h++) {
					ctx.fillStyle = "rgb(" + layer.outData[j] * 255 + "," + layer.outData[j] * 255 + "," + layer.outData[j] * 255 + ")";
					ctx.fillRect(
						x + 5 + (i + 1) * scale * SPREAD,
						y + 5 + h * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.outSize() * scale * 2) / 2),
						scale,
						scale
					);
				}

				for (var j = 0; j < layer.inSize(); j++) {
					ctx.fillStyle = "rgb(" + layer.inData[j] * 255 + "," + layer.inData[j] * 255 + "," + layer.inData[j] * 255 + ")";
					ctx.fillRect(
						x + 5 + i * scale * SPREAD,
						y + 5 + j * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2),
						scale,
						scale
					);
				}
				//end of render neurons for each layer
			}
			if (layer.constructor.name == "FCLayer") {
				//render weights in the fc layer

				for (var j = 0; j < layer.inSize(); j++) {
					for (var h = 0; h < layer.outSize(); h++) {
						ctx.strokeStyle =
							"rgb(" + -(layer.w[j * layer.outSize() + h] * 255) + "," + layer.w[j * layer.outSize() + h] * 255 + ",0)";
						ctx.beginPath();
						ctx.moveTo(
							x + 5 + i * scale * SPREAD + scale / 2,
							scale / 2 + y + 5 + j * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2)
						);
						ctx.lineTo(
							x + 5 + (i + 1) * scale * SPREAD + scale / 2,
							scale / 2 + y + 5 + h * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.outSize() * scale * 2) / 2)
						);
						ctx.stroke();
					} //end of foor loop each outSize
				} //end of for loop each inSize
			} //end of if layer is fc
			if (
				layer.constructor.name == "SigmoidLayer" ||
				layer.constructor.name == "SineLayer" ||
				layer.constructor.name == "TanhLayer" ||
				layer.constructor.name == "ReluLayer"
			) {
				//render the word sigmoid in between the nodes
				ctx.fillStyle = "white";
				ctx.font = scale + "px serif";
				ctx.fillText(
					layer.constructor.name.slice(0, -5),
					x + 5 + (i + 0.4) * scale * SPREAD,
					y + 5 + (layer.inSize() / 2) * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2)
				);
			}

			if ((layer.constructor.name = "ConvLayer")) {
			}
		} //end of for loop each layer
	};

	Ment.clamp = clamp;
	Ment.render = render;
	Ment.renderBox = renderBox;
	Ment.getLoss = getLoss;
	Ment.tanh = Math.tanh;
	Ment.lin = lin;
	Ment.bounce = bounce;
	Ment.isBrowser = isBrowser;
	Ment.gaussRandom = gaussRandom;
	Ment.polluteGlobal = polluteGlobal;
	Ment.protectNaN = protectNaN;
	Ment.inputError = inputError;
	Ment.printNet = printNet;
	let globalObject = isBrowser() ? window : global;
	if (globalObject.GPU) {
		const gpu = new GPU();
		Ment.gpu = gpu;
	}

	if (!Ment.isBrowser()) {
		//test if we are in node
		if (typeof module === "undefined" || typeof module.exports === "undefined") {
			window.Ment = Ment; // in ordinary browser attach library to window
		} else {
			module.exports = Ment; // in nodejs
		}
	}
}
