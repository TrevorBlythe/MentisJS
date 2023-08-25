{
	/*
if filter is 3x3 with indepth of 3 the first filter stored as such:
this.filterw = [0,1,2,3,4,5,6,7,8,
								0,1,2,3,4,5,6,7,8,
								0,1,2,3,4,5,6,7,8.......]

Each filter has depth equal to the InDepth.

this.filterw
[filterNum * this.inDepth * filterWidth * filterHeight + x + (y * filterWidth) + (depth * filterWidth * filterHeight)]

if you wanna input an image do it like this.
[r,r,r,r,g,g,g,g,b,b,b,b,b]
NOT like this:
[r,g,b,r,g,b,r,g,b,r,g,b]
Im sorry but I had to choose one 
*/

	class ConvLayer {
		static averageOutCosts = false; //probs
		static averageOutGrads = false; //ehhh
		constructor(inDim, filterDim, filters = 3, stride = 1, bias = true) {
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
			this.filters = filters; //the amount of filters
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.filterw = new Float64Array(filters * inDepth * filterWidth * filterHeight);
			this.filterws = new Float64Array(filters * inDepth * filterWidth * filterHeight);
			this.outData = new Float64Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			this.inData = new Float64Array(inWidth * inHeight * inDepth);
			this.accessed = new Float64Array(this.inData.length); //to average out the costs
			this.createAccessedMap();
			this.inData.fill(0); //to prevent mishap
			this.costs = new Float64Array(inWidth * inHeight * inDepth);
			this.b = new Float64Array(this.outData.length);
			this.bs = new Float64Array(this.outData.length);
			this.useBias = bias;
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Conv layer error: filters cannot be bigger than the input";
			}
			//init random weights
			for (var i = 0; i < this.filterw.length; i++) {
				this.filterw[i] = 0.5 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
			}
			for (var a = 0; a < 3; a++) {
				let newFilterw = this.filterw.slice(0);
				for (var f = 0; f < this.filters; f++) {
					for (var d = 0; d < this.inDepth; d++) {
						for (var x = 0; x < this.filterWidth; x++) {
							for (var y = 0; y < this.filterHeight; y++) {
								let count = 0;
								let ind = [f * this.inDepth * filterWidth * filterHeight + x + y * filterWidth + d * filterWidth * filterHeight];
								let indR = [
									f * this.inDepth * filterWidth * filterHeight + (x + 1) + y * filterWidth + d * filterWidth * filterHeight,
								];
								let indL = [
									f * this.inDepth * filterWidth * filterHeight + (x - 1) + y * filterWidth + d * filterWidth * filterHeight,
								];
								let indD = [
									f * this.inDepth * filterWidth * filterHeight + x + (y + 1) * filterWidth + d * filterWidth * filterHeight,
								];
								let indU = [
									f * this.inDepth * filterWidth * filterHeight + x + (y - 1) * filterWidth + d * filterWidth * filterHeight,
								];
								if (x < filterWidth - 1) count += this.filterw[indR];
								if (x > 1) count += this.filterw[indL];
								if (y < filterHeight - 1) count += this.filterw[indD];
								if (y > 1) count += this.filterw[indU];
								newFilterw[ind] += count / 5;
							}
						}
					}
				}
				this.filterw = newFilterw;
			}
			if (this.useBias) {
				for (var i = 0; i < this.b.length; i++) {
					this.b[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
			} else {
				this.b.fill(0);
			}
			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//DONT CHANGE THESE OR BIG BREAKY BREAKY!

			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
			this.wIHID = this.inWidth * this.inHeight * this.inDepth;
			this.fWIH = this.filterWidth * this.filterHeight;
			this.fWIHID = this.inDepth * this.filterHeight * this.filterWidth;
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
				this.filters,
			];
		}

		forward(input) {
			if (input) {
				if (input.length != this.inSize()) {
					throw inputError(this, input);
				}
				for (var i = 0; i < input.length; i++) {
					this.inData[i] = input[i];
				}
			}
			const outData = this.outData;
			const inData = this.inData;
			const biases = this.b;
			const filterw = this.filterw;
			const filterWidth = this.filterWidth;
			const filterHeight = this.filterHeight;
			const inWidth = this.inWidth;
			const inDepth = this.inDepth;
			const wIH = this.wIH;
			const fWIH = this.fWIH;
			const stride = this.stride;
			outData.fill(0);
			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * stride;
						outData[odi] += biases[odi];
						for (var h = 0; h < inDepth; h++) {
							const hWIH = h * wIH + ba;
							const hFWIH = h * fWIH + iFWIHID;
							for (var j = 0; j < filterHeight; j++) {
								const jGAIWBA = (j + ga) * inWidth + hWIH;
								const jFWHFWIH = j * filterWidth + hFWIH;
								for (var k = 0; k < filterWidth; k++) {
									outData[odi] += inData[k + jGAIWBA] * filterw[k + jFWHFWIH];
								}
							}
						}
					}
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			this.costs.fill(0); //reset the costs
			const inData = this.inData;
			const filterw = this.filterw;
			const filterws = this.filterws;
			const filterWidth = this.filterWidth;
			const filterHeight = this.filterHeight;
			const inWidth = this.inWidth;
			const inDepth = this.inDepth;
			const wIH = this.wIH;
			const fWIH = this.fWIH;
			const stride = this.stride;
			const costs = this.costs;

			//-----------------------------Beginning of monstrosity-----------------
			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * stride;
						for (var h = 0; h < inDepth; h++) {
							const hWIH = h * wIH;
							const hFWIH = h * fWIH + iFWIHID;
							for (var j = 0; j < filterHeight; j++) {
								const jGAIWBA = (j + ga) * inWidth + hWIH + ba;
								const jFWHFWIH = j * filterWidth + hFWIH;
								for (var k = 0; k < filterWidth; k++) {
									costs[k + jGAIWBA] += filterw[k + jFWHFWIH] * err[odi];
									filterws[k + jFWHFWIH] += inData[k + jGAIWBA] * err[odi];
								}
							}
						}
					}
				}
			}
			//---------------------------------End of monstrosity-----------------
			for (var i = 0; i < this.outSize(); i++) {
				this.bs[i] += err[i];
			}
			if (ConvLayer.averageOutCosts) {
				for (var i = 0; i < costs.length; i++) {
					costs[i] /= this.accessed[i];
				}
			}
		}

		createAccessedMap() {
			//This function creates an array the same size as the input image.
			//The values will represent the amount of times a kernel touches that particular pixel.
			//This is good to compute before hand so you can save on compute time.
			//We need these values for the sole purpose of averaging out the error of the input image
			this.accessed.fill(0);
			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * stride;
						for (var h = 0; h < inDepth; h++) {
							const hWIH = h * wIH;
							const hFWIH = h * fWIH + iFWIHID;
							for (var j = 0; j < filterHeight; j++) {
								const jGAIWBA = (j + ga) * inWidth + hWIH + ba;
								const jFWHFWIH = j * filterWidth + hFWIH;
								for (var k = 0; k < filterWidth; k++) {
									this.accessed[k + jGAIWBA]++;
								}
							}
						}
					}
				}
			}
		}

		getParamsAndGrads(forUpdate = true) {
			if (forUpdate) {
				if (ConvLayer.averageOutGrads) {
					for (var i = 0; i < this.filterws.length; i++) {
						this.filterws[i] /= this.filters;
					}
				}
			}
			if (this.useBias) {
				return [this.filterw, this.filterws, this.b, this.bs];
			} else {
				return [this.filterw, this.filterws];
			}
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "filterws" ||
					key == "filterbs" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "accessed" ||
					key == "bs" ||
					key == "ws" ||
					key == "hMFHPO" ||
					key == "wMFWPO" ||
					key == "hMFWMF" ||
					key == "wIH" ||
					key == "wIHID" ||
					key == "fWIH" ||
					key == "fWIHID" ||
					key == (this.useBias ? null : "b")
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ConvLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.filters,
				saveObject.stride,
				saveObject.useBias
			);
			for (var i = 0; i < layer.filterw.length; i++) {
				layer.filterw[i] = saveObject.filterw[i];
			}
			if (saveObject.useBias) {
				for (var i = 0; i < layer.b.length; i++) {
					layer.b[i] = saveObject.b[i];
				}
			}
			return layer;
		}
	}

	Ment.ConvLayer = ConvLayer;
	Ment.Conv = ConvLayer;
}
