{
	class ReluLayer {
		static leakySlope = 0.25;

		constructor(size) {
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(size);
			this.outData = new Float32Array(size);
			this.costs = new Float32Array(size); //costs for each neuron
		}

		forward(inData) {
			if (inData) {
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = this.inData[h] > 0 ? this.inData[h] : 0;
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
				if (this.outData[j] >= 0) {
					this.costs[j] = err;
				} else {
					this.costs[j] = err * swag.LeakyReluLayer.leakySlope;
				}
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

	swag.ReluLayer = ReluLayer;
	swag.Relu = ReluLayer;
}
