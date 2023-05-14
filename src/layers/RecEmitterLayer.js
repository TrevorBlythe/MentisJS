{
	class RecEmitterLayer {
		//Glorified Identity layer, only difference it has an ID and reference to the receiver with same ID
		constructor(id) {
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(0); //the inData
			this.outData = new Float32Array(0); //will be init when "connect" is called.
			this.costs = new Float32Array(0); //costs for each neuron
			this.receiver; // a reference to the receiver layer so we can skip layers
			//this will be set by the receiver  when the net is initialized
			this.savedOutData;
			this.pl = undefined; //the previous layer except interally its called "pl" so i can set getters and setters for the value "previousLayer"
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			if (layer.constructor.name == "RecReceiverLayer" && layer.id == this.id) {
				throw "You can't put a RecReceiver right before its corressponding RecEmitter. (because it doesnt know the output size) (try adding a dummy layer inbetween and giving it a size)";
			}
			this.inData = new Float32Array(layer.outSize());
			this.costs = new Float32Array(layer.outSize());
			this.pl = layer;

			this.outData = new Float32Array(layer.outSize());
			this.savedOutData = new Float32Array(layer.outSize());
			this.savedOutData.fill(0);
		}

		forward(inData) {
			//first save what was last outputted for the receiver
			if (this.where == "behind") {
				//this layer is to the left of receiver
				for (var i = 0; i < this.outSize(); i++) {
					this.savedOutData[i] = this.outData[i];
				}
			}
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
			if (this.where == "in front") {
				//this layer is to the right
				for (var i = 0; i < this.outSize(); i++) {
					this.savedOutData[i] = this.outData[i];
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			for (var j = 0; j < this.outData.length; j++) {
				this.costs[j] = (err[j] + this.costsFromReceiver[j]) / 2;
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
					key == "bs" ||
					key == "ws" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "emitter" ||
					key == "costsFromReceiver"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			//todo check if this is needed
			delete this.savedInSize;
			delete this.savedOutSize;

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new RecEmitterLayer(saveObject.id);
			return layer;
		}
	}

	Ment.RecEmitterLayer = RecEmitterLayer;
	Ment.RecEmitter = RecEmitterLayer;
	Ment.RecE = RecEmitterLayer;
}
