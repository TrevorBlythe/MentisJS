{
	class SineLayer {
		sine(z) {
			return Math.sin(z);
		}

		sinePrime(z) {
			return Math.cos(z);
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
					throw (
						'INPUT SIZE WRONG ON Sine LAYER:\nexpected size (' +
						this.inSize +
						'), got: (' +
						inData.length +
						')'
					);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = this.sine(this.inData[h]);
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
				this.costs[j] = err * this.sinePrime(this.inData[j]);
				loss += Math.pow(err, 2);
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

	swag.SineLayer = SineLayer;
	swag.Sine = SineLayer;
	swag.Sin = SineLayer;
}
