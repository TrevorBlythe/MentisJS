{
	class SigmoidLayer {
		sigmoid(z) {
			return 1 / (1 + Math.exp(-z));
		}

		sigmoidPrime(z) {
			return Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2);
		}

		constructor(size) {
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(size);
			this.outData = new Float32Array(size);
			this.costs = new Float32Array(size); //costs for each neuron
		}

		forward(inData) {
			if (inData) {
				if (inData.length != this.inData.length) {
					throw 'INPUT SIZE WRONG ON SIG LAYER:\nexpected size (' + this.inSize + '), got: (' + inData.length + ')';
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = this.sigmoid(this.inData[h]);
			}
			//Oh the misery
		}

		backward(expected) {
			let loss = 0;
			if (!expected) {
				if (this.nextLayer == undefined) {
					throw 'nothing to backpropagate!';
				}
				expected = [];
				for (var i = 0; i < this.outData.length; i++) {
					expected.push(this.nextLayer.costs[i] + this.nextLayer.inData[i]);
				}
			}

			for (var j = 0; j < this.outSize(); j++) {
				let err = expected[j] - this.outData[j];
				loss += Math.pow(err, 2);
				this.costs[j] = err * this.sigmoidPrime(this.inData[j]);
			}
			return loss / this.outSize();
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}
	}

	swag.SigmoidLayer = SigmoidLayer;
	swag.Sigmoid = SigmoidLayer;
	swag.Sig = SigmoidLayer;
}
