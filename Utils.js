
var swag = {};

{



  
	var isBrowser = new Function(
		'try {return this===window;}catch(e){ return false;}'
	);
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

	function sigmoid(z) {
		return 1 / (1 + Math.exp(-z));
	}

	function sigmoidPrime(z) {
		return Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2);
	}

	let getLoss = function (results, expected) {
		var loss = 0;
		for (var i = 0; i < results.length; i++) {
			loss += Math.pow(results[i] - expected[i], 2);
		}
		loss = loss / results.length;
		return loss;
	};
	let getLosses = function (results, expected) {
		var losses = [];
		for (var i = 0; i < results.length; i++) {
			losses.push(Math.pow(results[i] - expected[i], 2));
		}
		return losses;
	};
	
	let render = function (net, ctx, x, y, scale) {
		// a built in network renderer
		const SPREAD = 20;
		const DRAWBACKGROUND = true;
		let maxSize = 1;
		for (var i = 0; i < net.layers.length; i++) {
			if (net.layers[i].size > maxSize) {
				maxSize = net.layers[i].size;
			} //end of if
		} //end of for
		if (DRAWBACKGROUND) {
			ctx.fillStyle = 'grey';
			ctx.fillRect(
				x,
				y,
				5 + (net.layers.length - 1) * scale * SPREAD + scale,
				5 + maxSize * scale * 2
			);
		}

		for (i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];
			for (var j = 0; j < layer.size; j++) {
				ctx.fillStyle =
					'rgb(' +
					layer.activations[j] * 255 +
					',' +
					layer.activations[j] * 255 +
					',' +
					layer.activations[j] * 255 +
					')';
				ctx.fillRect(
					x + 5 + i * scale * SPREAD,
					y +
						5 +
						j * scale * 2 +
						((maxSize * scale * 2) / 2 - (layer.size * scale * 2) / 2),
					scale,
					scale
				);
				for (var h = 0; h < layer.w.length / layer.size; h++) {
					ctx.strokeStyle =
						'rgb(' +
						-(
							layer.w[j * (layer.w.length / layer.size) + h] *
							(255 / net.minmax)
						) +
						',' +
						layer.w[j * (layer.w.length / layer.size) + h] *
							(255 / net.minmax) +
						',0)';
					ctx.beginPath();
					ctx.moveTo(
						x + 5 + i * scale * SPREAD + scale / 2,
						scale / 2 +
							y +
							5 +
							j * scale * 2 +
							((maxSize * scale * 2) / 2 - (layer.size * scale * 2) / 2)
					);
					ctx.lineTo(
						x + 5 + (i + 1) * scale * SPREAD + scale / 2,
						scale / 2 +
							y +
							5 +
							h * scale * 2 +
							((maxSize * scale * 2) / 2 -
								(net.layers[i + 1].size * scale * 2) / 2)
					);
					ctx.stroke();
				}
			} //end of for loop
		} //end of for loop
	};
	
	
	swag.clamp = clamp;
	swag.render = render;
	swag.getLoss = getLoss;
	swag.tanh = Math.tanh;
	swag.sigmoidPrime = sigmoidPrime;
	swag.lin = lin;
	swag.bounce = bounce;
	swag.isBrowser = isBrowser;
	swag.sigmoid = sigmoid;
  
  

	if (!swag.isBrowser()) {
		//test if we are in node
		if (
			typeof module === 'undefined' ||
			typeof module.exports === 'undefined'
		) {
			window.swag = swag; // in ordinary browser attach library to window
		} else {
			module.exports = swag; // in nodejs
		}
	}
}
