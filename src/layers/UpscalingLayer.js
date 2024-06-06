{
	class UpscalingLayer {
		constructor(inDim, scale) {
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

			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.outWidth = inWidth * scale;
			this.outHeight = inHeight * scale;
			this.scale = scale;
			this.nextLayer; //the connected layer
			this.previousLayer; //the previousLayer
			this.inData = new Float64Array(inWidth * inHeight * inDepth); //the inData
			this.outData = new Float64Array(this.outWidth * this.outHeight * inDepth);
			this.grads = new Float64Array(this.inData.length); //grads for each activation in "inData"
		}
		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [this.outWidth, this.outHeight, this.inDepth];
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

			for (var i = 0; i < this.inDepth; i++) {
				for (var h = 0; h < this.inHeight * this.scale; h++) {
					for (var j = 0; j < this.inWidth * this.scale; j++) {
						this.outData[i * this.outHeight * this.outWidth + h * this.outWidth + j] =
							this.inData[
								Math.floor(i) * this.inHeight * this.inWidth +
									Math.floor(h / this.scale) * this.inWidth +
									Math.floor(j / this.scale)
							];
					}
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.grads;
			}
			this.grads.fill(0);

			for (var i = 0; i < this.inDepth; i++) {
				//this can be optimized
				for (var h = 0; h < this.inHeight * this.scale; h++) {
					for (var j = 0; j < this.inWidth * this.scale; j++) {
						let t = err[i * this.outHeight * this.outWidth + h * this.outWidth + j];
						this.grads[
							Math.floor(i) * this.inHeight * this.inWidth +
								Math.floor(h / this.scale) * this.inWidth +
								Math.floor(j / this.scale)
						] += t;
					}
				}
			}
			// for(var i = 0;i<this.grads.length;i++){
			// 	this.grads[i] /= this.scale;
			// }
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "ws" ||
					key == "bs" ||
					key == "inData" ||
					key == "netObject" ||
					key == "outData" ||
					key == "grads" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
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
			let layer = new UpscalingLayer([saveObject.inWidth, saveObject.inHeight, saveObject.inDepth], saveObject.scale);
			return layer;
		}
	}

	Ment.UpscalingLayer = UpscalingLayer;
	Ment.Upscaling = UpscalingLayer;
	Ment.Upscale = UpscalingLayer;
	Ment.UpScale = UpscalingLayer;
	Ment.UpScaling = UpscalingLayer;
	Ment.UpScalingLayer = UpscalingLayer;
}
