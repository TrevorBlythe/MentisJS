{
	class MinPoolLayer {
		constructor(inDim, filterDim, stride = 1) {
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

			if (filterDim.length != 2) {
				throw (
					this.constructor.name +
					" parameter error: Missing filter dimensions parameter. \n" +
					"First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)"
				);
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];

			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.outData = new Float64Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.inDepth
			);
			this.inData = new Float64Array(inWidth * inHeight * inDepth);
			this.grads = new Float64Array(inWidth * inHeight * inDepth);
			this.minIndexes = new Float64Array(this.outData.length);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Min Pool layer error: Pooling size (width / height) cannot be bigger than the inputs corresponding (width/height)";
			}

			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//Change these for broken code!
			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
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
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.inDepth,
			];
		}

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw Ment.inputError(this, inData);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}
			this.outData.fill(0);

			for (var g = 0; g < this.hMFHPO; g++) {
				const ga = g * this.stride;
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					const ba = b * this.stride;
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						const hWIH = h * this.wIH + ba;
						let min = this.inData[ga * this.inWidth + hWIH];
						for (var j = 0; j < this.filterHeight; j++) {
							const jGAIWBA = (j + ga) * this.inWidth + hWIH;
							for (var k = 0; k < this.filterWidth; k++) {
								if (this.inData[k + jGAIWBA] < min) {
									min = this.inData[k + jGAIWBA];
									this.minIndexes[odi] = k + jGAIWBA;
								}
							}
						}
						this.outData[odi] = min;
					}
				}
			}
		}

		backward(err) {
			this.grads.fill(0);
			if (!err) {
				err = this.nextLayer.grads;
			}
			for (var g = 0; g < this.hMFHPO; g++) {
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						this.grads[this.minIndexes[odi]] += err[odi];
					}
				}
			}
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
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

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, stride = 1,
			let saveObject = JSON.parse(json);
			let layer = new MinPoolLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.stride
			);
			return layer;
		}
	}

	Ment.MinPoolLayer = MinPoolLayer;
	Ment.MinPoolingLayer = MinPoolLayer;
	Ment.MinPool = MinPoolLayer;
}
