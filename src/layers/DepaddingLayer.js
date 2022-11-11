{
	class DepaddingLayer {
		constructor(outDim, pad) {
			if (outDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let outWidth = outDim[0];
			let outHeight = outDim[1];
			let outDepth = outDim[2];

			pad = pad || 2;
			this.outData = new Float32Array(outWidth * outHeight * outDepth);
			this.inData = new Float32Array((outWidth + pad * 2) * (outHeight + pad * 2) * outDepth);
			this.pad = pad;
			this.outWidth = outWidth;
			this.outHeight = outHeight;
			this.outDepth = outDepth;
			this.inData.fill(0);
			this.outData.fill(0);
			this.costs = new Float32Array(this.inData.length);
			this.costs.fill(0);
		}

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

			for (var i = 0; i < this.outDepth; i++) {
				for (var j = 0; j < this.outHeight; j++) {
					for (var h = 0; h < this.outWidth; h++) {
						let prop = i * this.outHeight * this.outWidth + j * this.outWidth + h;
						this.outData[prop] =
							this.inData[
								(j + 1) * this.pad * 2 +
									-this.pad +
									this.pad * (this.outWidth + this.pad * 2) +
									prop +
									i * ((this.outWidth + this.pad * 2) * this.pad) * 2 +
									i * (this.outHeight * this.pad) * 2
							];
					}
				}
			}
		}

		backward(expected) {
			let loss = 0;
			this.costs.fill(0);

			let geterr = (ind) => {
				return expected[ind] - this.outData[ind];
			};
			if (!expected) {
				geterr = (ind) => {
					return this.nextLayer.costs[ind];
				};
				if (this.nextLayer == undefined) {
					throw "error backproping on an unconnected layer with no expected parameter input";
				}
			}

			for (var i = 0; i < this.outDepth; i++) {
				for (var j = 0; j < this.outHeight; j++) {
					for (var h = 0; h < this.outWidth; h++) {
						let prop = i * this.outHeight * this.outWidth + j * this.outWidth + h;
						let err = geterr(prop);
						loss += Math.pow(err, 2);
						this.costs[
							(j + 1) * this.pad * 2 +
								-this.pad +
								this.pad * (this.outWidth + this.pad * 2) +
								prop +
								i * ((this.outWidth + this.pad * 2) * this.pad) * 2 +
								i * (this.outHeight * this.pad) * 2
						] = err;
					}
				}
			}
			return loss;
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		outSizeDimensions() {
			return [this.outWidth, this.outHeight, this.outDepth];
		}

		inSizeDimensions() {
			return [this.outWidth + this.pad * 2, this.outHeight + this.pad * 2, this.outDepth];
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
					key == "pl" ||
					key == "accessed"
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
			let layer = new DepaddingLayer([saveObject.outWidth, saveObject.outHeight, saveObject.outDepth], saveObject.pad);
			return layer;
		}
	}

	Ment.DepaddingLayer = DepaddingLayer;
	Ment.DepadLayer = DepaddingLayer;
	Ment.Depadding = DepaddingLayer;
	Ment.Depad = DepaddingLayer;
}
