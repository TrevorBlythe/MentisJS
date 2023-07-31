{
	class ResEmitterLayerGPU {
		//this layer outputs the inputs with no changes
		constructor(id) {
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.receiver; // a reference to the receiver layer so we can skip layers
			//this will be set by the receiver  when the net is initialized
			this.gpuCostsFromEmitterName; //gets set by the receiver
			this.pl = undefined;
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inSize = () => {
				return layer.outSize();
			};

			this.outSize = () => {
				return layer.outSize();
			};
			this.pl = layer;
		}

		// initGPUActivations(inDataGpuArrayName, outDataGpuArrayName, costsGpuArrayName, errorGpuArrayName) {
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

		// 	let temp = new Float32Array(this.inSize());
		// 	temp.fill(0);
		// 	Ment.webMonkeys.set(this.gpuInDataName, temp);
		// 	Ment.webMonkeys.set(this.gpuCostsArrayName, temp);

		// 	temp = new Float32Array(this.outSize());
		// 	temp.fill(0);
		// 	Ment.webMonkeys.set(this.gpuOutDataName, temp);
		// 	Ment.webMonkeys.set(this.gpuErrorArrayName, temp);

		// 	this.receiver.gpuInDataFromEmitterName = this.gpuOutDataName;
		// }

		forward() {
			//the outData of this layer is the same object referenced in the inData of the Receiver layer

			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				//this code should only run if user is dumb, but just in case
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			Ment.webMonkeys.work(
				this.inSize(),
				`
float act = (${this.gpuErrorArrayName}(i) + ${this.gpuCostsFromEmitterName}(i)) / 2.0;

;

${this.gpuCostsArrayName}(i) := act;
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
		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "receiver" ||
					key == "pl" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "gpuCostsFromEmitter" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "emitter"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ResEmitterLayerGPU(saveObject.id);
			return layer;
		}
	}

	Ment.ResEmitterLayerGPU = ResEmitterLayerGPU;
	Ment.ResEmitterGPU = ResEmitterLayerGPU;
	Ment.ResEGPU = ResEmitterLayerGPU;
}
