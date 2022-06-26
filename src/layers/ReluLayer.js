{
	class ReluLayer extends Ment.ActivationBase {
		constructor(size) {
			super(size);
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
				}
			}
			return loss / this.outSize();
		}
		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ReluLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.ReluLayer = ReluLayer;
	Ment.Relu = ReluLayer;
}
