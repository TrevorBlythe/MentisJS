//Base layer for all activations to inherit from

{
	class ActivationBaseGPU {
		constructor(size) {
			this.nextLayer; //the connected layer
			this.pl; //reference to previous layer
			this.inSize = () => {
				return size;
			};

			this.outSize = () => {
				return size;
			};
		}

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			//will get initialized berore initGPUactivations is run.
			this.inSize = () => {
				return layer.outSize();
			};

			this.outSize = () => {
				return layer.outSize();
			};
			this.pl = layer;
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "netObject" ||
					key == "outData" ||
					key == "grads" ||
					key == "gpuBiasName" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuGradsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "gpuBiasGradsName" ||
					key == "gpuWeightsName" ||
					key == "gpuWeightsGradsName" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;

			return ret;
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}
	}

	Ment.ActivationBaseGPU = ActivationBaseGPU;
}
