{
	class ResReceiverLayer {
		//this layer outputs the inputs with no changes
		constructor(id) {
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.inData; //the inData
			this.outData; //will be init when "onConnect" is called.
			this.costs; //costs for each neuron
			this.receiver; // a reference to the receiver layer so we can skip layers
			this.inDataFromEmitter;
			this.costsForEmitter;
			this.pl; // holds a reference to previous layer
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			console.log('test');
			this.inData = new Float32Array(layer.outSize());
			this.costs = new Float32Array(layer.outSize());
			this.pl = layer;
			//time to find this layers soulmate
			let found = false;
			let currentLayer = layer; //start at this layer go forward until find a reciever with the same ID
			while (!found) {
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.ResEmitter) {
					found = true;
				} else {
					currentLayer = currentLayer.previousLayer;
					if (currentLayer == undefined) {
						throw 'Could not find Matching Emitter Layer for Receiver Layer ID: ' + this.id;
					}
				}
			}
			this.emitter = currentLayer;
			this.inDataFromEmitter = this.emitter.outData;
			currentLayer.receiver = this; //so they can find each other again :)
			this.outData = new Float32Array(layer.outSize() + this.emitter.outSize());
			this.costsForEmitter = new Float32Array(this.emitter.outSize());
		}

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw 'INPUT SIZE WRONG ON (Input or output or linear) LAYER:\nexpected size (' + this.inSize() + '), got: (' + inData.length + ')';
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.inData.length; h++) {
				this.outData[h] = this.inData[h];
			}
			for (var h = this.inData.length; h < this.inData.length + this.inDataFromEmitter.length; h++) {
				this.outData[h] = this.inDataFromEmitter[h - this.inData.length];
			}
		}

		backward(expected) {
			let loss = 0;
			if (!expected) {
				if (this.nextLayer == undefined) {
					throw 'nothing to backpropagate!';
				}
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = this.nextLayer.costs[i];
					loss += this.costs[i];
				}
				for (var i = this.inData.length; i < this.inData.length + this.inDataFromEmitter.length; i++) {
					this.costsForEmitter[i - this.inData.length] = this.nextLayer.costs[i];
					loss += this.costsForEmitter[i - this.inData.length];
				}
			} else {
				for (var j = 0; j < this.inData.length; j++) {
					let err = expected[j] - this.outData[j];
					this.costs[j] = err;
					loss += Math.pow(err, 2);
				}
				for (var i = this.inData.length; i < this.inData.length + this.inDataFromEmitter.length; i++) {
					this.costsForEmitter[i - this.inData.length] = expected[i] - this.outData[i];
					loss += this.costsForEmitter[i - this.inData.length];
				}
			}
			return loss / this.inSize();
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
				if (key == 'inData' || key == 'outData' || key == 'costs' || key == 'nextLayer' || key == 'previousLayer') {
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
			let layer = new ResReceiverLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.ResReceiverLayer = ResReceiverLayer;
	Ment.ResReceiver = ResReceiverLayer;
	Ment.Receiver = ResReceiverLayer;
}
