{
	class DeconvParamLayer {
		constructor(inDim, filterDim, filters = 3, stride = 1, useBias = true, filterInputSize = 3, filterNetDepth = 1) {
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
			this.useBias = useBias;

			this.filters = filters; //the amount of filters
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterInputSize = filterInputSize;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.inData = new Float64Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			this.outData = new Float64Array(inWidth * inHeight * inDepth);
			this.inData.fill(0); //to prevent mishap
			this.outData.fill(0); //to prevent mishap
			this.grads = new Float64Array(this.inData.length);
			this.b = new Float64Array(this.outData.length);
			this.bs = new Float64Array(this.outData.length);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Conv layer error: filters cannot be bigger than the input";
			}

			this.filterWArrays = [];
			// this.filterWArrays.push(filterArchitecture);
			// if (this.filterWArrays[0].outSize != filterWidth * filterHeight * inDepth) {
			// 	let layer = this.filterWArrays[0];
			// 	throw `Filter architecture wrong for layer ${this.constructor.name + " " + this.id} : ${
			// 		layer.outSizeDimensions ? "," + layer.outSizeDimensions() : layer.outSize
			// 	} but expected size ${filterWidth * filterHeight * inDepth}`;
			// }
			// this.filterWArrays[0].errArr = new Float64Array(filterWidth * filterHeight * inDepth);
			for (var i = 0; i < filters; i++) {
				var filterNetLayers = [];
				filterNetLayers.push(new Ment.FCLayer(filterInputSize, filterWidth * filterHeight * inDepth));
				for (var j = 1; j < filterNetDepth; j++) {
					filterNetLayers.push(new Ment.Tanh()); //tanh so the generated weights are between -1 and 1
					filterNetLayers.push(new Ment.FCLayer(filterWidth * filterHeight * inDepth, filterWidth * filterHeight * inDepth));
				}
				this.filterWArrays.push(new Ment.Net(filterNetLayers, new Ment.SGD()));
				this.filterWArrays[i].errArr = new Float64Array(filterWidth * filterHeight * inDepth);
				this.filterWArrays[i].learningRate = 0.0;
				//the training/learning will be handled by the parent layer so we set the learning rate to 0
				//the parent layer will act as a proxy to the params/grads with the
				//getParamsAndGrads function
				this.filterWArrays[i].batchSize = Infinity;
			}

			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
			this.wIHID = this.inWidth * this.inHeight * this.inDepth;
			this.fWIH = this.filterWidth * this.filterHeight;
			this.fWIHID = this.inDepth * this.filterHeight * this.filterWidth;
		}

		forwardFilter(input) {
			for (var i = 0; i < this.filters; i++) {
				// console.log(this.filterWArrays[i].outData);
				this.filterWArrays[i].forward(input);
			}
		}

		setFilterFeild(feild, value) {
			for (var i = 0; i < this.filters; i++) {
				this.filterWArrays[i][feild] = value;
			}
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

		forward(input) {
			if (input) {
				if (input.length != this.inSize()) {
					throw Ment.inputError(this, input);
				}
				for (var i = 0; i < input.length; i++) {
					this.inData[i] = input[i];
				}
			}

			this.outData.fill(0);
			const inData = this.inData;
			const outData = this.outData;
			const filterWidth = this.filterWidth;
			const filterHeight = this.filterHeight;
			const inWidth = this.inWidth;
			const inDepth = this.inDepth;
			const wIH = this.wIH;
			const fWIH = this.fWIH;
			const stride = this.stride;

			//-------------Beginning of monstrosity-----------------
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
							const hWIH = h * wIH + ba;
							const hFWIH = h * fWIH; // + iFWIHID;
							for (var j = 0; j < filterHeight; j++) {
								const jGAIWBA = (j + ga) * inWidth + hWIH;
								const jFWHFWIH = j * filterWidth + hFWIH;
								for (var k = 0; k < filterWidth; k++) {
									outData[k + jGAIWBA] += inData[odi] * this.filterWArrays[i].outData[k + jFWHFWIH];
								}
							}
						}
					}
				}
			}
			//-------------End of monstrosity-----------------

			for (var i = 0; i < outData.length; i++) {
				outData[i] += this.b[i];
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.grads;
			}
			this.grads.fill(0); //reset the grads

			const grads = this.grads;
			const inData = this.inData;
			const filterWidth = this.filterWidth;
			const filterHeight = this.filterHeight;
			const inWidth = this.inWidth;
			const inDepth = this.inDepth;
			const wIH = this.wIH;
			const fWIH = this.fWIH;
			const stride = this.stride;
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
							const hFWIH = h * fWIH; // + iFWIHID;
							for (var j = 0; j < filterHeight; j++) {
								const jGAIWBA = (j + ga) * inWidth + hWIH + ba;
								const jFWHFWIH = j * filterWidth + hFWIH;
								for (var k = 0; k < filterWidth; k++) {
									grads[odi] += this.filterWArrays[i].layers[0].outData[k + jFWHFWIH] * err[k + jGAIWBA];
									this.filterWArrays[i].errArr[k + jFWHFWIH] += inData[odi] * err[k + jGAIWBA];
								}
							}
						}
					}
				}
			}

			//backprop all the filters
			for (var i = 0; i < this.filters; i++) {
				this.filterWArrays[i].backward(this.filterWArrays[i].errArr, false, true);
				//reset error
				// console.log(this.filterWArrays[i].err);
				this.filterWArrays[i].errArr.fill(0.0);
			}
			for (var i = 0; i < this.outData.length; i++) {
				this.bs[i] += err[i];
			}
		}

		save() {
			//FUTURE ME FIX THIS
			//It is saving useless information and taking up space
			// in the json file fix it l8er
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "filterws" ||
					key == "filterbs" ||
					key == "inData" ||
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
					key == (this.useBias ? null : "b") ||
					key == "filterWArrays"
				) {
					if (key == "filterWArrays") {
						let rett = [];
						for (var i = 0; i < value.length; i++) {
							rett.push(value[i].save());
						}
						return rett;
					}
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, filters = 3, stride = 1,
			let saveObject = JSON.parse(json);
			// console.log();
			let layer = new DeconvParamLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.filters,
				saveObject.stride,
				saveObject.useBias,
				saveObject.filterInputSize
			);
			// console.log(saveObject.filterWArrays);
			if (saveObject.useBias) {
				for (var i = 0; i < layer.b.length; i++) {
					layer.b[i] = saveObject.b[i];
				}
			}
			for (var i = 0; i < layer.filterWArrays.length; i++) {
				layer.filterWArrays[i] = Ment.Net.load(saveObject.filterWArrays[i]);
				layer.filterWArrays[i].errArr = new Float64Array(saveObject.filterWidth * saveObject.filterHeight * saveObject.inDepth);
				layer.filterWArrays[i].learningRate = 0;
				layer.filterWArrays[i].batchSize = Infinity;
			}
			return layer;
		}
	}

	Ment.DeconvParamLayer = DeconvParamLayer;
	Ment.DeConvParamLayer = DeconvParamLayer;
	Ment.DeconvParam = DeconvParamLayer;
	Ment.DeConvParam = DeconvParamLayer;
}
