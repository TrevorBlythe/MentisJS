{
	class IdentityLayerGPU {
		//this layer outputs the inputs with no changes
		constructor(size) {
			this.nextLayer; //the connected layer
			this.pl; //reference to previous layer
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
			this.inSize = () => {
				return size;
			};

			this.outSize = () => {
				return size;
			};
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			// try to connect to it
			this.inSize = () => {
				return layer.outSize();
			};

			this.outSize = () => {
				return layer.outSize();
			};
			this.pl = layer;
		}
		forward() {
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
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			Ment.webMonkeys.work(
				this.inSize(),
				`
float act = ${this.gpuErrorArrayName}(i);

;

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "pl" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "nextLayer" ||
					key == "previousLayer"
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
			let layer = new IdentityLayerGPU(saveObject.savedSize);
			return layer;
		}
	}

	Ment.IdentityLayerGPU = IdentityLayerGPU;
	Ment.IdentityGPU = IdentityLayerGPU;
	Ment.InputGPU = IdentityLayerGPU;
	Ment.DummyLayerGPU = IdentityLayerGPU;
	Ment.DummyGPU = IdentityLayerGPU;
	Ment.InputLayerGPU = IdentityLayerGPU;
	Ment.OutputLayerGPU = IdentityLayerGPU;
	Ment.OutputGPU = IdentityLayerGPU;
}
