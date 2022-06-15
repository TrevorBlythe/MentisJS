var swag = swag || {};
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


	let getLoss = function (results, expected) {
		var loss = 0;
		for (var i = 0; i < results.length; i++) {
			loss += Math.pow(results[i] - expected[i], 2);
		}
		loss = loss / results.length;
		return loss;
	};

	
	let render = function (net, ctx, x, y, scale) {
		// a built in network renderer
		const SPREAD = 10;
		const DRAWBACKGROUND = true;
		let maxSize = 1;
		for (var i = 0; i < net.layers.length; i++) {
			if(net.layers[i].constructor.name != 'ConvLayer' && net.layers[i].constructor.name != 'MaxPoolLayer'){
				
			if (net.layers[i].inSize() > maxSize) {
				maxSize = net.layers[i].inSize();
			} //end of if
			if (net.layers[i].outSize() > maxSize) {
				maxSize = net.layers[i].outSize();
			} //end of if
		} //end of for
		if (DRAWBACKGROUND) {
			ctx.fillStyle = 'grey';
			ctx.fillRect(
				x,
				y,
				5 + (net.layers.length) * scale * SPREAD + scale,
				5 + maxSize * scale * 2
			);
		}
		}

//render the neurons in each layer...
		for (i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];
			if(layer.constructor.name != 'ConvLayer' && layer.constructor.name != 'MaxPoolLayer'){


			for (var h = 0; h < layer.outSize(); h++) {
				ctx.fillStyle =
				'rgb(' +
				layer.outData[j] * 255 +
				',' +
				layer.outData[j] * 255 +
				',' +
				layer.outData[j] * 255 +
				')';
			ctx.fillRect(
				x + 5 + (i+1) * scale * SPREAD,
				y +
					5 +
					h * scale * 2 +
					((maxSize * scale * 2) / 2 - (layer.outSize() * scale * 2) / 2),
				scale,
				scale
			);
		}
	
	
		for (var j = 0; j < layer.inSize(); j++) {
			ctx.fillStyle =
				'rgb(' +
				layer.inData[j] * 255 +
				',' +
				layer.inData[j] * 255 +
				',' +
				layer.inData[j] * 255 +
				')';
			ctx.fillRect(
				x + 5 + i * scale * SPREAD,
				y +
					5 +
					j * scale * 2 +
					((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2),
				scale,
				scale
			);
		}
//end of render neurons for each layer

	}
			if(layer.constructor.name == "FCLayer"){ //render weights in the fc layer
				
				for (var j = 0; j < layer.inSize(); j++) {
					for (var h = 0; h < layer.outSize(); h++) {
						ctx.strokeStyle =
							'rgb(' +
							-(
								layer.w[j * layer.outSize() + h] *
								(255)
							) +
							',' +
							layer.w[j * layer.outSize() + h] *
								(255) +
							',0)';
						ctx.beginPath();
						ctx.moveTo(
							x + 5 + i * scale * SPREAD + scale / 2,
							scale / 2 +
								y +
								5 +
								j * scale * 2 +
								((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2)
						);
						ctx.lineTo(
							x + 5 + (i + 1) * scale * SPREAD + scale / 2,
							scale / 2 +
								y +
								5 +
								h * scale * 2 +
								((maxSize * scale * 2) / 2 -
									(layer.outSize() * scale * 2) / 2)
						);
						ctx.stroke();
					}//end of foor loop each outSize
				} //end of for loop each inSize
		}//end of if layer is fc 
		if(layer.constructor.name == "SigmoidLayer" ||
			 layer.constructor.name == "SineLayer" || 
			 layer.constructor.name == "TanhLayer"){
			//render the word sigmoid in between the nodes
			ctx.fillStyle = "white";
			ctx.font = scale  + 'px serif';
			ctx.fillText(layer.constructor.name.slice(0,-5), 	x + 5 + (i + 0.4) * scale * SPREAD,
			y +
				5 +
				(layer.inSize()/2) * scale * 2 +
				((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2));
		}

		if(layer.constructor.name = "ConvLayer"){
			
		}

		} //end of for loop each layer
	};
	

	
	swag.clamp = clamp;
	swag.render = render;
	swag.getLoss = getLoss;
	swag.tanh = Math.tanh;
	swag.lin = lin;
	swag.bounce = bounce;
	swag.isBrowser = isBrowser;
	if(window.GPU){
		const gpu = new GPU();
		swag.gpu = gpu; 
	}
  

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
