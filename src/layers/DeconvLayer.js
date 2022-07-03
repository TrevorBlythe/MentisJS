{
	class DeconvLayer {
		static averageOutCosts = false;
		static averageOutGrads = false;
		constructor(
			inDim,
			filterDim, filters = 3, stride = 1) {


			if(inDim.length != 3){
				throw this.constructor.name + " parameter error: Missing dimensions parameter. \n"
				+ "First parameter in layer must be an 3 length array, width height and depth";
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			if(filterDim.length != 2){
				throw this.constructor.name + " parameter error: Missing filter dimensions parameter. \n"
				+ "First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)";
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];

			this.lr = 0.01; //learning rate, this will be set by the Net object

			this.filters = filters; //the amount of filters
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.filterw = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			this.filterws = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			this.trainIterations = 0;
			this.inData = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) *
					Math.ceil((inHeight - filterHeight + 1) / stride) *
					this.filters
			);
			this.outData = new Float32Array(inWidth * inHeight * inDepth);
			this.inData.fill(0); //to prevent mishap
			this.outData.fill(0); //to prevent mishap
			this.costs = new Float32Array(this.inData.length);
			this.b = new Float32Array(this.outData.length);
			this.bs = new Float32Array(this.outData.length);
				this.accessed = new Float32Array(this.inData.length).fill(0);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw 'Conv layer error: filters cannot be bigger than the input';
			}
			//init random weights
			for (var i = 0; i < this.filterw.length; i++) {
				this.filterw[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
			}
			for (var i = 0; i < this.b.length; i++) {
				this.b[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
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

		outSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		inSizeDimensions() {
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.filters,
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

			//-------------Beginning of monstrosity-----------------
			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * this.stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * this.stride;
						for (var h = 0; h < this.inDepth; h++) {
							const hWIH = h * this.wIH + ba;
							const hFWIH = h * this.fWIH + iFWIHID;
							for (var j = 0; j < this.filterHeight; j++) {
								const jGAIWBA = (j + ga) * this.inWidth + hWIH;
								const jFWHFWIH = j * this.filterWidth + hFWIH;
								for (var k = 0; k < this.filterWidth; k++) {
									this.outData[k + jGAIWBA] += this.inData[odi] * this.filterw[k + jFWHFWIH];
								}
							}
						}
					}
				}
			}
			//-------------End of monstrosity-----------------

			for (var i = 0; i < this.outData.length; i++) {
				this.outData[i] += this.b[i];
			}
		}

		backward(expected) {
			let loss = 0;
			this.trainIterations++;
			for (var i = 0; i < this.inSize(); i++) {
				//reset the costs
				this.costs[i] = 0;
			}

			if (!expected) {
				if (this.nextLayer == undefined) {
					throw 'error backproping on an unconnected layer with no expected parameter input deconv layer';
				}
			}

			let getCost = (ind) => {
				return this.nextLayer.costs[ind];
			};
			if (expected) {
				getCost = (ind) => {
					return expected[ind] - this.outData[ind];
				};
			}

			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * this.stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * this.stride;
						for (var h = 0; h < this.inDepth; h++) {
							const hWIH = h * this.wIH;
							const hFWIH = h * this.fWIH + iFWIHID;
							for (var j = 0; j < this.filterHeight; j++) {
								const jGAIWBA = (j + ga) * this.inWidth + hWIH + ba;
								const jFWHFWIH = j * this.filterWidth + hFWIH;
								for (var k = 0; k < this.filterWidth; k++) {
									let err = getCost(k + jGAIWBA);
									loss += Math.pow(err, 2);

									this.costs[odi] += this.filterw[k + jFWHFWIH] * err;

									// this.accessed[odi]++;

									this.filterws[k + jFWHFWIH] += this.inData[odi] * err;
								}
							}
						}
					}
				}
			}

			for (var i = 0; i < this.outData.length; i++) {
				this.bs[i] += getCost(i);
			}
			if(DeconvLayer.averageOutCosts){
				for (var i = 0; i < this.inSize(); i++) {
					this.costs[i] = this.costs[i] / (this.accessed[i] > 0 ? this.accessed[i] : 1);
					this.accessed[i] = 0;
				}
			}

			return loss / (this.wMFWPO * this.hMFHPO * this.filters);
		}

		updateParams(optimizer) {
			for (var i = 0; i < this.filterw.length; i++) {
			if(DeconvLayer.averageOutGrads){
				this.filterws[i] /= this.outSize() / this.filters;
			}
				this.filterws[i] /= this.trainIterations;
				this.filterws[i] = Ment.protectNaN(this.filterws[i]);
				this.filterw[i] += Math.max(-this.lr, Math.min(this.lr, this.filterws[i] * this.lr));

				this.filterws[i] = 0;
			}
			for (var i = 0; i < this.b.length; i++) {
				this.b[i] += this.bs[i] * this.lr;
				this.bs[i] = 0;
			}
			this.trainIterations = 0;
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == 'filterws' ||
					key == 'filterbs' ||
					key == 'inData' ||
					key == 'outData' ||
					key == 'costs' ||
					key == 'gpuEnabled' ||
					key == 'trainIterations' ||
					key == 'nextLayer' ||
					key == 'previousLayer' ||
					key == 'pl'
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, filters = 3, stride = 1,
			let saveObject = JSON.parse(json);
			let layer = new DeconvLayer(
				[saveObject.inWidth,
				saveObject.inHeight,
				saveObject.inDepth],
				[saveObject.filterWidth,
				saveObject.filterHeight],
				saveObject.filters,
				saveObject.stride,
				saveObject.useBias
			);
			for (var i = 0; i < layer.filterw.length; i++) {
				layer.filterw[i] = saveObject.filterw[i];
			}
			if (saveObject.b) {
				for (var i = 0; i < layer.b.length; i++) {
					layer.b[i] = saveObject.b[i];
				}
			}
			layer.lr = saveObject.lr;
			return layer;
		}
	}

	Ment.DeconvLayer = DeconvLayer;
	Ment.DeConvLayer = DeconvLayer;
	Ment.Deconv = DeconvLayer;
	Ment.DeConv = DeconvLayer;

	//only uncomment if you know what your doing
}
