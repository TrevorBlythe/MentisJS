{
	/*
Only difference is the filter depths are 1 and they only go over one depth of the input
*/

	class DepthWiseConvLayer {
		constructor(inDim, filterDim, filters = 3, stride = 1, bias = false) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}

			if (filters % inDim[2] != 0) {
				throw "DepthwiseConvLayer error: The amount of filters must be a multiple of the input depth";
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
			this.filterw = new Float64Array(filters * filterWidth * filterHeight);
			this.filterws = new Float64Array(filters * filterWidth * filterHeight);
			this.outData = new Float64Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			this.inData = new Float64Array(inWidth * inHeight * inDepth);
			this.inData.fill(0); //to prevent mishap
			this.grads = new Float64Array(inWidth * inHeight * inDepth);
			this.b = new Float64Array(this.outData.length);
			this.bs = new Float64Array(this.outData.length);
			this.useBias = bias;
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Conv layer error: filters cannot be bigger than the input";
			}

			for (var i = 0; i < this.filterw.length; i++) {
				this.filterw[i] = 3 * (Math.random() - 0.5);
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
			//note to self i changed this so it might be big breaky breaky
			this.fWIHID = this.filterHeight * this.filterWidth;
		}

		reInitializeFilter(index) {
			for (var i = this.fWIHID * index; i < this.fWIHID * index + this.fWIHID; i++) {
				this.filterw[i] = 3 * (Math.random() - 0.5);
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
				const iFWIHID = i * this.fWIH;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * stride;
						outData[odi] += biases[odi];
						// for (var h = 0; h < inDepth; h++) {
						//since this is depthwise the depth we operate on will depend on the index of the filter
						const h = i % inDepth;
						const hWIH = h * wIH + ba;
						const hFWIH = iFWIHID;
						for (var j = 0; j < filterHeight; j++) {
							const jGAIWBA = (j + ga) * inWidth + hWIH;
							const jFWHFWIH = j * filterWidth + hFWIH;
							for (var k = 0; k < filterWidth; k++) {
								outData[odi] += inData[k + jGAIWBA] * filterw[k + jFWHFWIH];
							}
						}
						// }
					}
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.grads;
			}
			this.grads.fill(0); //reset the grads
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
			const grads = this.grads;

			//-----------------------------Beginning of monstrosity-----------------
			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIH;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * stride;
						// for (var h = 0; h < inDepth; h++) {
						const h = i % inDepth;
						const hWIH = h * wIH;
						const hFWIH = iFWIHID;
						for (var j = 0; j < filterHeight; j++) {
							const jGAIWBA = (j + ga) * inWidth + hWIH + ba;
							const jFWHFWIH = j * filterWidth + hFWIH;
							for (var k = 0; k < filterWidth; k++) {
								grads[k + jGAIWBA] += filterw[k + jFWHFWIH] * err[odi];
								filterws[k + jFWHFWIH] += inData[k + jGAIWBA] * err[odi];
							}
						}
						// }
					}
				}
			}
			//---------------------------------End of monstrosity-----------------
			for (var i = 0; i < this.outSize(); i++) {
				this.bs[i] += err[i];
			}
		}

		getParamsAndGrads(forUpdate = true) {
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
					key == "netObject" ||
					key == "netObject" ||
					key == "outData" ||
					key == "grads" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
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
			let layer = new DepthWiseConvLayer(
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

	Ment.DepthWiseConvLayer = DepthWiseConvLayer;
	Ment.DepthWiseConv = DepthWiseConvLayer;
	Ment.DWConv = DepthWiseConvLayer;
}
