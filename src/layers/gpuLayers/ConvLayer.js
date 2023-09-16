{
	class ConvLayerGPU {
		//yeah these averaging things wont do antyhing cuz i havent implemented them yet
		//dont worry the layer will still werk
		//Proggramming this layer right here gave me schizofrinia!!!
		static averageOutCosts = true; //probs
		static averageOutGrads = true; //ehhh
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
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;

			this.gpuFilterW = Ment.makeid(8); //new Float32Array(filters * inDepth * filterWidth * filterHeight);
			let temp = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			for (var i = 0; i < filters * inDepth * filterWidth * filterHeight; i++) {
				temp[i] = (1 * Math.random() * (Math.random() > 0.5 ? -1 : 1)) / this.filters; //set random weights
			}
			// //this next for loop gives the starting weights a random "pattern" using cellular automata.
			// //since most people are not gonna be training on random noise, having patters probably helps
			// //train it faster
			// for (var a = 0; a < 3; a++) {
			// 	let newFilterw = temp.slice(0);
			// 	for (var f = 0; f < this.filters; f++) {
			// 		for (var d = 0; d < this.inDepth; d++) {
			// 			for (var x = 0; x < this.filterWidth; x++) {
			// 				for (var y = 0; y < this.filterHeight; y++) {
			// 					let count = 0;
			// 					let ind = [f * this.inDepth * filterWidth * filterHeight + x + y * filterWidth + d * filterWidth * filterHeight];
			// 					let indR =
			// 						f * this.inDepth * filterWidth * filterHeight + (x + 1) + y * filterWidth + d * filterWidth * filterHeight;
			// 					let indL =
			// 						f * this.inDepth * filterWidth * filterHeight + (x - 1) + y * filterWidth + d * filterWidth * filterHeight;
			// 					let indD =
			// 						f * this.inDepth * filterWidth * filterHeight + x + (y + 1) * filterWidth + d * filterWidth * filterHeight;
			// 					let indU =
			// 						f * this.inDepth * filterWidth * filterHeight + x + (y - 1) * filterWidth + d * filterWidth * filterHeight;
			// 					if (x < filterWidth - 1) count += this.filterw[indR];
			// 					if (x > 1) count += this.filterw[indL];
			// 					if (y < filterHeight - 1) count += this.filterw[indD];
			// 					if (y > 1) count += temp[indU];
			// 					newFilterw[ind] += count / 5;
			// 				}
			// 			}
			// 		}
			// 	}
			// 	temp = newFilterw;
			// }
			Ment.webMonkeys.set(this.gpuFilterW, temp);

			this.gpuFilterWGrads = Ment.makeid(8); //new Float32Array(filters * inDepth * filterWidth * filterHeight);
			Ment.webMonkeys.set(this.gpuFilterWGrads, filters * inDepth * filterWidth * filterHeight);

			// this.accessed = new Float32Array(this.inData.length).fill(0); //to average out the costs

			this.gpuBiasName = Ment.makeid(8); //new Float32Array(this.outData.length);
			this.gpuBiasGradsName = Ment.makeid(8); //new Float32Array(this.outData.length);
			this.useBias = bias;
			temp = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			if (this.useBias) {
				for (
					var i = 0;
					i < Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters;
					i++
				) {
					temp[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1); // set random bias
				}
			}
			Ment.webMonkeys.set(this.gpuBiasName, temp);
			Ment.webMonkeys.set(
				this.gpuBiasGradsName,
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Conv layer error: filters cannot be bigger than the input";
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

			this.gpuAccessedMap = Ment.makeid(8); //a map with the same size of the input times filter each input pixel will store what filter pixels can reach it. Needed cuz gpu is so so dumb unlike cpu

			//initialize it.
			// prettier-ignore
			{
			let tempAccessed = new Float32Array(inWidth * inHeight * filterWidth * filterHeight);

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
									let filterInd = j * this.filterWidth + k;
									let indice = (k + jGAIWBA)*this.filterHeight * this.filterWidth + filterInd;
									tempAccessed[indice] = g * this.wMFWPO + b + 1.1;
								}
							}
						}
					}
				}
			}

			Ment.webMonkeys.set(this.gpuAccessedMap, tempAccessed);


		}
		}

		inSize() {
			return this.inWidth * this.inHeight * this.inDepth;
		}

		outSize() {
			return (
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride) *
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride) *
				this.filters
			);
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

		forward() {
			Ment.webMonkeys.work(
				this.filters * this.hMFHPO * this.wMFWPO,
				`
int ifromi = int(i / ${this.hMFHPO * this.wMFWPO});
int gfromi = int((i - ifromi * ${this.hMFHPO * this.wMFWPO}) / ${this.wMFWPO});
int bfromi = int((i - ifromi * ${this.hMFHPO * this.wMFWPO} - gfromi * ${this.wMFWPO}));


int iHMFWMF = ifromi * ${this.hMFWMF};
int iFWIHID = ifromi * ${this.fWIHID};
int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};
float act = ${this.gpuBiasName}(odi);

for (int h = 0; h < ${this.inDepth}; h++) {
	int hWIH = h * ${this.wIH} + ba;
	int hFWIH = h * ${this.fWIH} + iFWIHID;

	for (int j = 0; j < ${this.filterHeight}; j++) {
		int jGAIWBA = (j + ga) * ${this.inWidth} + hWIH;
		int jFWHFWIH = j * ${this.filterWidth} + hFWIH;
		for (int k = 0; k < ${this.filterWidth}; k++) {
			act += ${this.gpuInDataName}(k + jGAIWBA + ${this.gpuInDataStartIndex}) * ${this.gpuFilterW}(k + jFWHFWIH);
		}
	}
}

;

${this.gpuOutDataName}(odi + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}
		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			//v will be height c will be width h will be depth
			//this took forever to think of
			Ment.webMonkeys.work(
				this.inWidth * this.inHeight * this.inDepth,
				`
float act = 0.0;
int remaining = i;
int hfromi = int(remaining / (${this.inWidth * this.inHeight}));
remaining = remaining - (hfromi * ${this.inWidth * this.inHeight});
int vfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (vfromi * ${this.inWidth});
int cfromi = remaining;
int cdi = i;
int acc = ((i - hfromi * ${this.inWidth * this.inHeight}) * ${this.filterHeight * this.filterWidth});

for (int jfromi = 0; jfromi < ${this.filterHeight}; jfromi++) {
for (int kfromi = 0; kfromi < ${this.filterWidth}; kfromi++) {
if(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi) > 0.5){

int amnt = int(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi)) - 1;
int gfromi = int(amnt / ${this.wMFWPO});
int bfromi = amnt - gfromi * ${this.wMFWPO};
for (int ifromi = 0; ifromi < ${this.filters}; ifromi++) {
int iHMFWMF = ifromi * ${this.hMFWMF};
int iFWIHID = ifromi * ${this.fWIHID};

int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};
int hWIH = hfromi * ${this.wIH};
int hFWIH = hfromi * ${this.fWIH} + iFWIHID;
int jFWHFWIH = jfromi * ${this.filterWidth} + hFWIH;

act += ${this.gpuFilterW}(kfromi + jFWHFWIH) * ${this.gpuErrorArrayName}(odi);

}



}
}
};

;

${this.gpuCostsArrayName}(cdi) := act;
`
			);

			Ment.webMonkeys.work(
				this.filters * this.inDepth * this.filterWidth * this.filterHeight,
				`
				int remaining = i;
				int ifromi = int(remaining / (${this.inDepth * this.filterHeight * this.filterWidth}));
remaining = remaining - (ifromi * ${this.inDepth * this.filterHeight * this.filterWidth});
int hfromi = int(remaining / (${this.filterHeight * this.filterWidth}));
remaining = remaining - (hfromi * ${this.filterHeight * this.filterWidth});
int jfromi = int(remaining / (${this.filterWidth}));
remaining = remaining - (jfromi * ${this.filterWidth});
int kfromi = remaining;


int cdi = i;

int iFWIHID = ifromi * ${this.fWIHID};
int hFWIH = hfromi * ${this.fWIH} + iFWIHID;
int jFWHFWIH = jfromi * ${this.filterWidth} + hFWIH;
float act = ${this.gpuFilterWGrads}(kfromi + jFWHFWIH);
for (int vfromi = 0; vfromi < ${this.inHeight}; vfromi++) {
for (int cfromi = 0; cfromi < ${this.inWidth}; cfromi++) {
int acc = (((vfromi * ${this.inWidth} + cfromi)) * ${this.filterHeight * this.filterWidth});
if(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi) > 0.5){

int amnt = int(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi)) - 1;
int gfromi = int(amnt / ${this.wMFWPO});
int bfromi = amnt - gfromi * ${this.wMFWPO};
int iHMFWMF = ifromi * ${this.hMFWMF};

int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};
int hWIH = hfromi * ${this.wIH};
int jGAIWBA = (jfromi + ga) * ${this.inWidth} + hWIH + ba;

act += ${this.gpuInDataName}(kfromi + jGAIWBA + ${this.gpuInDataStartIndex}) * ${this.gpuErrorArrayName}(odi) * 2.0;





}
}
};

;

${this.gpuFilterWGrads}(kfromi + jFWHFWIH) := act;
`
			);

			if (this.useBias) {
				Ment.webMonkeys.work(
					this.outSize(),
					`
float act = ${this.gpuBiasGradsName}(i) + ${this.gpuErrorArrayName}(i);

;

${this.gpuBiasGradsName}(i) := act;
`
				);
			}
		}

		getParamsAndGrads(forUpdate = true) {
			if (this.useBias) {
				return [this.gpuFilterW, this.gpuFilterWGrads, this.gpuBiasName, this.gpuBiasGradsName];
			} else {
				return [this.gpuFilterW, this.gpuFilterWGrads];
			}
		}

		save() {
			this.filterw = Ment.webMonkeys.get(this.gpuFilterW);
			this.b = Ment.webMonkeys.get(this.gpuBiasName);
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "filterws" ||
					key == "gpuFilterWGrads" ||
					key == "gpuBiasGradsName" ||
					key == "gpuAccessedMap" ||
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
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
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

			delete this.filterw;
			delete this.b;
			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ConvLayerGPU(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.filters,
				saveObject.stride,
				saveObject.useBias
			);
			let temp = new Float32Array(layer.filters * layer.inDepth * layer.filterWidth * layer.filterHeight);
			for (var i = 0; i < layer.filters * layer.inDepth * layer.filterWidth * layer.filterHeight; i++) {
				temp[i] = saveObject.filterw[i];
			}
			Ment.webMonkeys.set(layer.gpuFilterW, temp);
			if (layer.useBias) {
				temp = new Float32Array(layer.outSize());
				for (var i = 0; i < layer.outSize(); i++) {
					temp[i] = saveObject.b[i];
				}
				Ment.webMonkeys.set(layer.gpuBiasName, temp);
			}
			return layer;
		}
	}

	Ment.ConvLayerGPU = ConvLayerGPU;
	Ment.ConvGPU = ConvLayerGPU;
}
