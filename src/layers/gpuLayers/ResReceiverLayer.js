{
	class ResReceiverLayerGPU {
		//this layer outputs the inputs with no changes
		constructor(id, mode = "concat") {
			//mode can be concat or add or average
			this.mode = mode;
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.emitter;
			this.gpuGradsForEmitter;
			this.pl; // holds a reference to previous layer
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inSize = () => {
				return layer.outSize();
			};

			this.pl = layer;

			//time to find this layers soulmate
			let found = false;
			let currentLayer = layer; //start at this layer go backward until find a emitter with the same ID
			while (!found) {
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.ResEmitterGPU) {
					found = true;
				} else {
					currentLayer = currentLayer.previousLayer;
					if (currentLayer == undefined) {
						throw "Could not find Matching Emitter Layer for Receiver Layer ID: " + this.id;
					}
				}
			}
			this.emitter = currentLayer; //so they can find each other again :)
			currentLayer.receiver = this;
			if (this.mode == "add" || this.mode == "average") {
				if (layer.outSize() != this.emitter.outSize()) {
					throw "emitter size must equal the size of the previous layer of the corresponding receiver layer";
				}
				this.outSize = () => {
					return layer.outSize();
				};
			} else if (this.mode == "concat") {
				this.outSize = () => {
					return layer.outSize() + this.emitter.outSize();
				};
			}
			this.gpuGradsForEmitterName = Ment.makeid(8);
			Ment.webMonkeys.set(this.gpuGradsForEmitterName, this.emitter.outSize());
			this.emitter.gpuGradsFromEmitterName = this.gpuGradsForEmitterName;
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		//we dont look in front because residual data only goes forwards.

		forward() {
			//the outData of this layer is the same object referenced in the inData of the Receiver layer

			if (this.mode == "concat") {
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
				);
				Ment.webMonkeys.work(
					this.emitter.outSize(),
					`
float act = ${this.emitter.gpuOutDataName}(i + ${this.emitter.gpuOutDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.inSize()} + ${this.gpuOutDataStartIndex}) := act;
				`
				);
			} else if (this.mode == "add") {
				Ment.webMonkeys.work(
					this.outSize(),
					`
float act = ${this.emitter.gpuOutDataName}(i + ${this.emitter.gpuOutDataStartIndex}) + ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
				);
			} else if (this.mode == "average") {
				Ment.webMonkeys.work(
					this.outSize(),
					`
float act = (${this.emitter.gpuOutDataName}(i + ${this.emitter.gpuOutDataStartIndex}) + ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex})) / 2.0;

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
				);
			}
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			if (this.mode == "concat") {
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i);

;

${this.gpuGradsArrayName}(i) := act;
				`
				);
				Ment.webMonkeys.work(
					this.emitter.outSize(),
					`
float act = ${this.gpuErrorArrayName}(i + ${this.inSize()});

;

${this.gpuGradsForEmitterName}(i) := act;				`
				);
			} else if (this.mode == "add") {
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i) / 2.0;

;

${this.gpuGradsArrayName}(i) := act;
				`
				);
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i) / 2.0;

;

${this.gpuGradsForEmitterName}(i) := act;				`
				);
			} else if (this.mode == "average") {
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i);

;

${this.gpuGradsArrayName}(i) := act;
				`
				);
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i);

;

${this.gpuGradsForEmitterName}(i) := act;				`
				);
			}
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "receiver" ||
					key == "pl" ||
					key == "inData" ||
					key == "netObject" ||
					key == "outData" ||
					key == "grads" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "gpuGradsForEmitter" ||
					key == "gpuInDataFromEmitterName" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuGradsArrayName" ||
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
			let layer = new ResReceiverLayerGPU(saveObject.id, saveObject.mode);
			return layer;
		}
	}

	Ment.ResReceiverLayerGPU = ResReceiverLayerGPU;
	Ment.ResReceiverGPU = ResReceiverLayerGPU;
	Ment.ResRGPU = ResReceiverLayerGPU;
}
