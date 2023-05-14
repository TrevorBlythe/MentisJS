{
	class AveragePoolLayer {
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
			this.outData = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.inDepth
			);
			this.inData = new Float32Array(inWidth * inHeight * inDepth);
			this.costs = new Float32Array(inWidth * inHeight * inDepth);
			this.maxIndexes = new Float32Array(this.outData.length);
			this.accessed = new Float32Array(this.costs.length).fill(1);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Average Pool layer error: Pooling size (width / height) cannot be bigger than the inputs corresponding (width/height)";
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

			for (var g = 0; g < this.hMFHPO; g++) {
				const ga = g * this.stride;
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					const ba = b * this.stride;
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						const hWIH = h * this.wIH + ba;
						let avg = 0;
						for (var j = 0; j < this.filterHeight; j++) {
							const jGAIWBA = (j + ga) * this.inWidth + hWIH;
							for (var k = 0; k < this.filterWidth; k++) {
								avg += this.inData[k + jGAIWBA];
							}
						}
						this.outData[odi] = avg / (this.filterWidth * this.filterHeight);
					}
				}
			}
		}

		backward(err) {
			this.costs.fill(0);
			if (!err) {
				err = this.nextLayer.costs;
			}
			for (var g = 0; g < this.hMFHPO; g++) {
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						this.costs[this.maxIndexes[odi]] += err[odi];
						this.accessed[this.maxIndexes[odi]]++;
					}
				}
			}
			for (var g = 0; g < this.hMFHPO; g++) {
				const ga = g * this.stride;
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					const ba = b * this.stride;
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						const hWIH = h * this.wIH + ba;
						for (var j = 0; j < this.filterHeight; j++) {
							const jGAIWBA = (j + ga) * this.inWidth + hWIH;
							for (var k = 0; k < this.filterWidth; k++) {
								this.costs[k + jGAIWBA] += err[odi];
							}
						}
					}
				}
			}
			for (var i = 0; i < this.costs.length; i++) {
				this.costs[i] /= this.accessed[i] + this.filterWidth * this.filterHeight; //Average the grad
				this.accessed[i] = 0;
			}
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
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
			let layer = new AveragePoolLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.stride
			);
			return layer;
		}
	}

	Ment.AveragePoolLayer = AveragePoolLayer;
	Ment.AveragePool = AveragePoolLayer;
	Ment.AveragePooling = AveragePoolLayer;
	Ment.AvgPool = AveragePoolLayer;
	Ment.AvgPoolLayer = AveragePoolLayer;
	Ment.AvgPoolingLayer = AveragePoolLayer;
}
