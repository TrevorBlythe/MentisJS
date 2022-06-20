{
	/*
if you wanna input an image do it like this.
[r,r,r,r,g,g,g,g,b,b,b,b,b]
NOT like this:
[r,g,b,r,g,b,r,g,b,r,g,b]

*/

	class MaxPoolLayer {
		constructor(inWidth, inHeight, inDepth, filterWidth, filterHeight, stride = 1, padding = 0) {
			if (padding != 0) {
				throw (
					'Dear user, I have not implemented padding yet.. set it to zero to avoid this message. ' +
					'Star the project if you want me to complete it. or send me 5 bucks ill do it right now.'
				);
			}
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.padding = 0; //havent implemented padding yet
			this.outData = new Float32Array(Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.inDepth);
			this.inData = new Float32Array(inWidth * inHeight * inDepth);
			this.costs = new Float32Array(inWidth * inHeight * inDepth);
			this.maxIndexes = new Float32Array(this.outData.length);
			this.accessed = new Float32Array(this.costs.length).fill(1);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw 'Max Pool layer error: Pooling size (width / height) cannot be bigger than the inputs corresponding (width/height)';
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
			return [Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride), Math.ceil((this.inHeight - this.filterHeight + +1) / this.stride), this.inDepth];
		}

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw (
						'INPUT SIZE WRONG ON MAX POOL LAYER:\nexpected array size (' +
						this.inSize() +
						', dimensions: [' +
						this.inSizeDimensions() +
						']), got: (' +
						inData.length +
						')'
					);
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
						let max = this.inData[ga * this.inWidth + hWIH];
						for (var j = 0; j < this.filterHeight; j++) {
							const jGAIWBA = (j + ga) * this.inWidth + hWIH;
							for (var k = 1; k < this.filterWidth; k++) {
								if (this.inData[k + jGAIWBA] > max) {
									max = this.inData[k + jGAIWBA];
									this.maxIndexes[odi] = k + jGAIWBA;
								}
							}
						}
						this.outData[odi] = max;
					}
				}
			}
		}

		backward(expected) {
			this.costs.fill(0);
			let loss = 0;

			if (!expected) {
				// -- sometimes the most effiecant way is the least elagant one...
				if (this.nextLayer == undefined) {
					throw 'error backproping on an unconnected layer with no expected parameter input';
				}
			}

			for (var g = 0; g < this.hMFHPO; g++) {
				const ga = g * this.stride;
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					const ba = b * this.stride;
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						let err = !expected ? this.nextLayer.costs[odi] : expected[odi] - this.outData[odi];
						loss += Math.pow(err, 2);
						this.costs[this.maxIndexes[odi]] += err;
						this.accessed[this.maxIndexes[odi]]++;
					}
				}
			}

			// for (var i = 0; i < this.inSize(); i++) {
			// 	this.costs[i] = this.costs[i] / (this.accessed[i] > 0 ? this.accessed[i] : 1);
			// 	this.accessed[i] = 0;
			// }
			return loss / (this.hMFHPO * this.wMFWPO * this.inDepth);
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == 'inData' ||
					key == 'outData' ||
					key == 'costs' ||
					key == 'gpuEnabled' ||
					key == 'trainIterations' ||
					key == 'nextLayer' ||
					key == 'previousLayer'
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, stride = 1, padding = 0
			let saveObject = JSON.parse(json);
			let layer = new MaxPoolLayer(
				saveObject.inWidth,
				saveObject.inHeight,
				saveObject.inDepth,
				saveObject.filterWidth,
				saveObject.filterHeight,
				saveObject.stride,
				saveObject.padding
			);
			return layer;
		}
	}

	Ment.MaxPoolLayer = MaxPoolLayer;
	Ment.MaxPool = MaxPoolLayer;
}
