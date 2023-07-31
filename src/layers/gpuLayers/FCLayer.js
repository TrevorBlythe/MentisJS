{
	class FCLayerGPU {
		constructor(inSize, outSize, useBias) {
			this.useBias = useBias == undefined ? true : useBias;
			this.gpuWeightsName = Ment.makeid(8); //generates random 8 character string
			this.gpuWeightsGradsName = Ment.makeid(8); //generates random 8 character string
			this.gpuBiasName = Ment.makeid(8); //generates random 8 character string
			this.gpuBiasGradsName = Ment.makeid(8); //generates random 8 character string
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
			this.inSize = () => {
				return inSize;
			};

			this.outSize = () => {
				return outSize;
			};

			let temp = new Float32Array(inSize * outSize);
			for (var i = 0; i < inSize * outSize; i++) {
				temp[i] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1); //set random weights
			}
			Ment.webMonkeys.set(this.gpuWeightsName, temp);
			temp.fill(0);
			Ment.webMonkeys.set(this.gpuWeightsGradsName, temp);

			temp = new Float32Array(outSize);
			if (this.useBias) {
				for (var i = 0; i < outSize; i++) {
					temp[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1); // set random bias
				}
			}
			Ment.webMonkeys.set(this.gpuBiasName, temp);
			temp.fill(0);
			Ment.webMonkeys.set(this.gpuBiasGradsName, temp);
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
			Ment.webMonkeys.work(
				this.inSize(),
				`
float act = 0.0;
for (int j = 0; j < ${this.outSize()}; j++) {
	act += ${this.gpuWeightsName}(j + i * ${this.outSize()}) * ${this.gpuErrorArrayName}(j) * 2.0;
}
act = act / ${this.outSize()}.0;

;

${this.gpuCostsArrayName}(i) := act;
				`
			);

			//this code is so weird but works :\
			Ment.webMonkeys.work(
				this.inSize() * this.outSize(),
				`
float act = 0.0;
int j = i - (int(i / ${this.outSize()}) * ${this.outSize()});
int k = (i - j) / ${this.outSize()};
act = ${this.gpuInDataName}(k + ${this.gpuInDataStartIndex}) * ${this.gpuErrorArrayName}(j) * 2.0;
act += ${this.gpuWeightsGradsName}(i);

;

${this.gpuWeightsGradsName}(i) := act;
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
				return [this.gpuWeightsName, this.gpuWeightsGradsName, this.gpuBiasName, this.gpuBiasGradsName];
			} else {
				return [this.gpuWeightsName, this.gpuWeightsGradsName];
			}
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			this.savedInSize = this.inSize();
			this.savedOutSize = this.outSize();
			this.w = Ment.webMonkeys.get(this.gpuWeightsName);
			this.b = Ment.webMonkeys.get(this.gpuBiasName);
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "outData" ||
					key == "inData" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "gpuBiasName" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "gpuBiasGradsName" ||
					key == "gpuWeightsName" ||
					key == "gpuWeightsGradsName" ||
					key == (this.useBias ? null : "b")
				) {
					return undefined;
				}

				return value;
			});

			delete this.savedInSize;
			delete this.w;
			delete this.b;
			delete this.savedOutSize;

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new FCLayerGPU(saveObject.savedInSize, saveObject.savedOutSize, saveObject.useBias);

			let temp = new Float32Array(layer.inSize() * layer.outSize());
			for (var i = 0; i < layer.inSize() * layer.outSize(); i++) {
				temp[i] = saveObject.w[i];
			}
			Ment.webMonkeys.set(layer.gpuWeightsName, temp);
			if (layer.useBias) {
				temp = new Float32Array(layer.outSize());
				for (var i = 0; i < layer.outSize(); i++) {
					temp[i] = saveObject.b[i];
				}
				Ment.webMonkeys.set(layer.gpuBiasName, temp);
			}
			return layer;
		}

		// initGPUActivations(
		// 	inDataGpuArrayName,
		// 	outDataGpuArrayName,
		// 	costsGpuArrayName,
		// 	errorGpuArrayName,
		// 	inDataGPUStartIndex,
		// 	outDataGPUStartIndex
		// ) {
		// 	//i know its confusing but costs are the erorr of the indata neurons and error is the error of the outdata neurons
		// 	if (!inDataGpuArrayName) {
		// 		this.gpuInDataName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!outDataGpuArrayName) {
		// 		this.gpuOutDataName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!costsGpuArrayName) {
		// 		this.gpuCostsArrayName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!errorGpuArrayName) {
		// 		this.gpuErrorArrayName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	this.gpuInDataName = inDataGpuArrayName;
		// 	this.gpuOutDataName = outDataGpuArrayName;
		// 	this.gpuCostsArrayName = costsGpuArrayName;
		// 	this.gpuErrorArrayName = errorGpuArrayName;
		// 	this.gpuInDataStartIndex = inDataGPUStartIndex;
		// 	this.gpuOutDataStartIndex = outDataGPUStartIndex;

		// 	let temp = new Float32Array(this.inSize());
		// 	temp.fill(0);
		// 	Ment.webMonkeys.set(this.gpuInDataName, temp);
		// 	Ment.webMonkeys.set(this.gpuCostsArrayName, temp);

		// 	temp = new Float32Array(this.outSize());
		// 	temp.fill(0);
		// 	Ment.webMonkeys.set(this.gpuOutDataName, temp);
		// 	Ment.webMonkeys.set(this.gpuErrorArrayName, temp);
		// }

		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = 0.0;
for (int j = 0; j < ${this.inSize()}; j++) {
	act += ${this.gpuInDataName}(j + ${this.gpuInDataStartIndex}) * ${this.gpuWeightsName}(i + j * ${this.outSize()});
}
act += ${this.gpuBiasName}(i);
${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
							`
			);
		}
	}

	Ment.FCLayerGPU = FCLayerGPU;
	Ment.FCGPU = FCLayerGPU;
}
