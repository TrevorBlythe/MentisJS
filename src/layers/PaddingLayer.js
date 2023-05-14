{
	class PaddingLayer {
		constructor(inDim, pad, padwith) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			pad = pad || 2;
			padwith = padwith | 0;
			this.inData = new Float32Array(inWidth * inHeight * inDepth);
			this.outData = new Float32Array((inWidth + pad * 2) * (inHeight + pad * 2) * inDepth);
			this.pad = pad;
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.padwith = padwith;
			this.outData.fill(padwith);
			this.inData.fill(0);
			this.costs = new Float32Array(this.inData.length);
			this.costs.fill(0);
		}

		// const handler = {
		// 	get: function (obj, prop) {
		// 		prop = Math.floor((prop % (inHeight * inWidth)) / inHeight + 1) * padding * 2 + -padding + padding * (inWidth + padding * 2) + parseFloat(prop);
		// 		return obj[prop];
		// 	},
		// 	set: function (obj, prop, value) {
		// 		prop = Math.floor((prop % (inHeight * inWidth)) / inHeight + 1) * padding * 2 + -padding + padding * (inWidth + padding * 2) + parseFloat(prop);
		// 		obj[prop] = value;
		// 	},
		// };

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw inputError(this, inData);
				}
				//Fun fact: the fastest way to copy an array in javascript is to use a For Loop
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}
			this.outData.fill(this.padwith);

			for (var i = 0; i < this.inDepth; i++) {
				for (var j = 0; j < this.inHeight; j++) {
					for (var h = 0; h < this.inWidth; h++) {
						let prop = i * this.inHeight * this.inWidth + j * this.inWidth + h;
						this.outData[
							(j + 1) * this.pad * 2 +
								-this.pad +
								this.pad * (this.inWidth + this.pad * 2) +
								prop +
								i * ((this.inWidth + this.pad * 2) * this.pad) * 2 +
								i * (this.inHeight * this.pad) * 2
						] = this.inData[prop];
					}
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			this.costs.fill(0);

			for (var i = 0; i < this.inDepth; i++) {
				for (var j = 0; j < this.inHeight; j++) {
					for (var h = 0; h < this.inWidth; h++) {
						let prop = i * this.inHeight * this.inWidth + j * this.inWidth + h;
						this.costs[prop] =
							err[
								(j + 1) * this.pad * 2 +
									-this.pad +
									this.pad * (this.inWidth + this.pad * 2) +
									prop +
									i * ((this.inWidth + this.pad * 2) * this.pad) * 2 +
									i * (this.inHeight * this.pad) * 2
							];
					}
				}
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [this.inWidth + this.pad * 2, this.inHeight + this.pad * 2, this.inDepth];
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new PaddingLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				saveObject.pad,
				saveObject.padwith
			);
			return layer;
		}
	}

	Ment.PaddingLayer = PaddingLayer;
	Ment.PadLayer = PaddingLayer;
	Ment.Padding = PaddingLayer;
	Ment.Pad = PaddingLayer;
}
