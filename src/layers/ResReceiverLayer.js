{
	class ResReceiverLayer {
		//this layer outputs the inputs with no changes
		constructor(id, mode = "concat") {
			//mode can be concat or add or average
			this.mode = mode;
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.inData; //the inData
			this.outData; //will be init when "onConnect" is called.
			this.grads; //grads for each neuron
			this.emitter;
			this.inDataFromEmitter;
			this.gradsForEmitter;
			this.pl; // holds a reference to previous layer
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inData = new Float64Array(layer.outSize());
			this.grads = new Float64Array(layer.outSize());
			this.pl = layer;
			//time to find this layers soulmate
			let found = false;
			let currentLayer = layer; //start at this layer go backward until find a emitter with the same ID
			while (!found) {
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.ResEmitter) {
					found = true;
				} else {
					currentLayer = currentLayer.previousLayer;
					if (currentLayer == undefined) {
						throw "Could not find Matching Emitter Layer for Receiver Layer ID: " + this.id;
					}
				}
			}
			this.emitter = currentLayer; //so they can find each other again :)
			this.inDataFromEmitter = this.emitter.outData;
			currentLayer.receiver = this;
			if (this.mode == "add" || this.mode == "average") {
				if (layer.outSize() != this.emitter.outSize()) {
					throw "emitter size must equal the size of the previous layer of the corresponding receiver layer";
				}
				this.outData = new Float64Array(layer.outSize());
			} else if (this.mode == "concat") {
				this.outData = new Float64Array(layer.outSize() + this.emitter.outSize());
			}
			this.gradsForEmitter = new Float64Array(this.emitter.outSize());
		}

		//we dont look in front because residual data only goes forwards.

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw Ment.inputError(this, inData);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}
			if (this.mode == "concat") {
				//first the data behind it and then the skip layer data
				for (var h = 0; h < this.inData.length; h++) {
					this.outData[h] = this.inData[h];
				}
				for (var h = this.inData.length; h < this.inData.length + this.inDataFromEmitter.length; h++) {
					this.outData[h] = this.inDataFromEmitter[h - this.inData.length];
				}
			} else if (this.mode == "add") {
				for (var i = 0; i < this.outData.length; i++) {
					this.outData[i] = this.inData[i] + this.inDataFromEmitter[i];
				}
			} else if (this.mode == "average") {
				for (var i = 0; i < this.outData.length; i++) {
					this.outData[i] = (this.inData[i] + this.inDataFromEmitter[i]) / 2;
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.grads;
			}

			if (this.mode == "concat") {
				//fixed a very atrocious error
				for (var i = 0; i < this.inData.length; i++) {
					this.grads[i] = err[i];
				}
				for (var i = this.inData.length; i < this.inData.length + this.inDataFromEmitter.length; i++) {
					this.gradsForEmitter[i - this.inData.length] = err[i];
				}
			} else if (this.mode == "add") {
				for (var i = 0; i < this.inData.length; i++) {
					this.grads[i] = err[i] / 2;
					this.gradsForEmitter[i] = err[i] / 2;
				}
			} else if (this.mode == "average") {
				for (var i = 0; i < this.inData.length; i++) {
					this.grads[i] = err[i] * 2;
					this.gradsForEmitter[i] = err[i] * 2;
				}
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "emitter" ||
					key == "pl" ||
					key == "receiver" ||
					key == "ws" ||
					key == "bs" ||
					key == "nl" ||
					key == "inData" ||
					key == "netObject" ||
					key == "outData" ||
					key == "grads" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "gradsForEmitter" ||
					key == "inDataFromEmitter"
				) {
					return undefined;
				}

				return value;
			});
			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ResReceiverLayer(saveObject.id, saveObject.mode);
			return layer;
		}
	}

	Ment.ResReceiverLayer = ResReceiverLayer;
	Ment.ResReceiver = ResReceiverLayer;
	Ment.ResR = ResReceiverLayer;
}
