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

	var inputError = function (layer, arr) {
		let ret = `INPUT SIZE WRONG ON ${layer.constructor.name}: `;
		ret += `expected size (${layer.inSize()}${layer.inSizeDimensions ? ',' + layer.inSizeDimensions() : ''}), got (${arr.length})`;
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
	var isBrowser = () => !(typeof window === 'undefined');
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

	let renderBox = function (net, ctx, x, y, scale, spread, background) {
		if (background == undefined) {
			background = 'white';
		}
		if (background == false) {
			background = 'rgba(0,0,0,0)';
		}
		const SPREAD = spread || 10;
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

		ctx.fillRect(x, y, scale + net.layers.length * scale * spread, scale + 6 * scale);
		for (var i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];
			let layerSize = layer.inSize() > layer.outSize() ? layer.inSize() : layer.outSize();
			layerSize += (maxSize - layerSize) / 3;
			let layerLeftSize = layer.inSize() + (maxSize - layer.inSize()) / 3;
			let layerRightSize = layer.outSize() + (maxSize - layer.outSize()) / 3;
			Ment.seed = layer.constructor.name.charCodeAt(0) * 5; //------ make random seed from layer name
			for (var n = 1; n < layer.constructor.name.length; n++) {
				Ment.seed += layer.constructor.name.charCodeAt(n);
			} //----------End of make random seed from layer name
			ctx.fillStyle = `rgb(${Ment.seededRandom(150, 255)},${Ment.seededRandom(100, 255)},${Ment.seededRandom(100, 255)})`;
			let xy = (spread * scale) / 2 + scale + i * scale * spread - scale + x;
			let yy = 5 + ((maxSize - layerLeftSize) / maxSize / 2) * scale * 6 + y;
			let xyy = (spread * scale) / 2 + scale + i * scale * spread - scale + x + scale;
			let yyy = 5 + ((maxSize - layerRightSize) / maxSize / 2) * scale * 6 + y;
			ctx.beginPath();
			ctx.moveTo(xy, yy);
			ctx.lineTo(xyy, yyy);
			ctx.lineTo(xyy, yyy + (layerRightSize / maxSize) * scale * 6);
			ctx.lineTo(xy, yy + (layerLeftSize / maxSize) * scale * 6);
			ctx.lineTo(xy, yy);
			ctx.fill();
			ctx.font = `${(layerSize / maxSize) * scale * (5 / layer.constructor.name.length)}px serif`;
			ctx.fillStyle = 'black';
			ctx.save();
			ctx.textAlign = 'center';
			ctx.translate(xy + (xyy - xy) / 2, yy + ((layerLeftSize / maxSize) * scale * 6) / 2);
			ctx.rotate(-Math.PI / 2);
			//(${layer.inSize()})(${layer.outSize()})
			ctx.fillText(layer.constructor.name, 0, 0);
			ctx.font = `${((layerSize / maxSize) * scale * (5 / layer.constructor.name.length)) / 2}px serif`;
			ctx.fillText(
				`(${layer.inSizeDimensions ? layer.inSizeDimensions() : layer.inSize()})(${layer.outSizeDimensions ? layer.outSizeDimensions() : layer.outSize()})`,
				0,
				((layerSize / maxSize) * scale * (5 / layer.constructor.name.length)) / 1.5
			);
			ctx.restore();
		}
	};

	let render = function (net, ctx, x, y, scale, background, spread) {
		// a built in network renderer
		if (background == undefined) {
			background = true;
		}
		const SPREAD = spread || 10;
		const DRAWBACKGROUND = background;
		let maxSize = 1;
		for (var i = 0; i < net.layers.length; i++) {
			if (net.layers[i].constructor.name != 'ConvLayer' && net.layers[i].constructor.name != 'MaxPoolLayer') {
				if (net.layers[i].inSize() > maxSize) {
					maxSize = net.layers[i].inSize();
				} //end of if
				if (net.layers[i].outSize() > maxSize) {
					maxSize = net.layers[i].outSize();
				} //end of if
			} //end of for
			if (DRAWBACKGROUND) {
				ctx.fillStyle = 'grey';
				ctx.fillRect(x, y, 5 + net.layers.length * scale * SPREAD + scale, 5 + maxSize * scale * 2);
			}
		}

		//render the neurons in each layer...
		for (i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];
			if (layer.constructor.name != 'ConvLayer' && layer.constructor.name != 'MaxPoolLayer') {
				for (var h = 0; h < layer.outSize(); h++) {
					ctx.fillStyle = 'rgb(' + layer.outData[j] * 255 + ',' + layer.outData[j] * 255 + ',' + layer.outData[j] * 255 + ')';
					ctx.fillRect(x + 5 + (i + 1) * scale * SPREAD, y + 5 + h * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.outSize() * scale * 2) / 2), scale, scale);
				}

				for (var j = 0; j < layer.inSize(); j++) {
					ctx.fillStyle = 'rgb(' + layer.inData[j] * 255 + ',' + layer.inData[j] * 255 + ',' + layer.inData[j] * 255 + ')';
					ctx.fillRect(x + 5 + i * scale * SPREAD, y + 5 + j * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2), scale, scale);
				}
				//end of render neurons for each layer
			}
			if (layer.constructor.name == 'FCLayer') {
				//render weights in the fc layer

				for (var j = 0; j < layer.inSize(); j++) {
					for (var h = 0; h < layer.outSize(); h++) {
						ctx.strokeStyle = 'rgb(' + -(layer.w[j * layer.outSize() + h] * 255) + ',' + layer.w[j * layer.outSize() + h] * 255 + ',0)';
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
				layer.constructor.name == 'SigmoidLayer' ||
				layer.constructor.name == 'SineLayer' ||
				layer.constructor.name == 'TanhLayer' ||
				layer.constructor.name == 'ReluLayer'
			) {
				//render the word sigmoid in between the nodes
				ctx.fillStyle = 'white';
				ctx.font = scale + 'px serif';
				ctx.fillText(
					layer.constructor.name.slice(0, -5),
					x + 5 + (i + 0.4) * scale * SPREAD,
					y + 5 + (layer.inSize() / 2) * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2)
				);
			}

			if ((layer.constructor.name = 'ConvLayer')) {
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
	let globalObject = isBrowser() ? window : global;
	if (globalObject.GPU) {
		const gpu = new GPU();
		Ment.gpu = gpu;
	}

	if (!Ment.isBrowser()) {
		//test if we are in node
		if (typeof module === 'undefined' || typeof module.exports === 'undefined') {
			window.Ment = Ment; // in ordinary browser attach library to window
		} else {
			module.exports = Ment; // in nodejs
		}
	}
}
