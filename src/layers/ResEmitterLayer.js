{
	class ResEmitterLayer {
		//this layer outputs the inputs with no changes
		constructor(id) {
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.inData = new Float64Array(0); //the inData
			this.outData = new Float64Array(0); //will be init when "connect" is called.
			this.grads = new Float64Array(0); //grads for each neuron
			this.receiver; // a reference to the receiver layer so we can skip layers
			//this will be set by the receiver  when the net is initialized
			this.pl = undefined;
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inData = new Float64Array(layer.outSize());
			this.grads = new Float64Array(layer.outSize());
			this.pl = layer;

			this.outData = new Float64Array(layer.outSize());
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

			for (var h = 0; h < this.outSize(); h++) {
				//the outData of this layer is the same object referenced in the inData of the Receiver layer
				this.outData[h] = this.inData[h];
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.grads;
			}
			this.grads.fill(0);

			for (var i = 0; i < this.outData.length; i++) {
				this.grads[i] += err[i];
				this.grads[i] += this.receiver.gradsForEmitter[i];
				this.grads[i] /= 2;
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
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
					key == "emitter"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;
			delete this.savedOutSize;

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ResEmitterLayer(saveObject.id);
			return layer;
		}
	}

	Ment.ResEmitterLayer = ResEmitterLayer;
	Ment.ResEmitter = ResEmitterLayer;
	Ment.ResE = ResEmitterLayer;
}
