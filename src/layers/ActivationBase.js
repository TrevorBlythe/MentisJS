//Base layer for all activations to inherit from

{
	class ActivationBase {
		constructor(size) {
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(size);
			this.outData = new Float32Array(size);
			this.costs = new Float32Array(size); //costs for each neuron
			this.pl; //reference to previous layer
		}

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			// try to connect to it
			if (this.inData.length == 0) {
				//if not already initialized
				this.inData = new Float32Array(layer.outSize());
				this.outData = new Float32Array(layer.outSize());
				this.costs = new Float32Array(layer.outSize());
			}
			this.pl = layer;
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		forward(inData, actFunction) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw Ment.inputError(this, inData);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = actFunction(this.inData[h]);
			}
			//Oh the misery
		}

		backward(expected, actFunctionPrime) {
			let loss = 0;
			if (!expected) {
				if (this.nextLayer == undefined) {
					throw "nothing to backpropagate!";
				}
				expected = [];
				for (var i = 0; i < this.outData.length; i++) {
					expected.push(this.nextLayer.costs[i] + this.nextLayer.inData[i]);
				}
			}

			for (var j = 0; j < this.outSize(); j++) {
				let err = expected[j] - this.outData[j];
				loss += Math.pow(err, 2);
				this.costs[j] = err * actFunctionPrime(this.inData[j]);
			}
			return loss / this.outSize();
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
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
	}

	Ment.ActivationBase = ActivationBase;
}
