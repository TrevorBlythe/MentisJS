{
	class LeakyReluLayer extends Ment.ActivationBase {
		static leakySlope = 0.25;

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
				this.outData[h] = this.inData[h] > 0 ? this.inData[h] : this.inData[h] * LeakyReluLayer.leakySlope;
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			for (var j = 0; j < this.outSize(); j++) {
				if (this.outData[j] >= 0) {
					this.costs[j] = err[j];
				} else {
					this.costs[j] = err[j] * LeakyReluLayer.leakySlope;
				}
			}
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new LeakyReluLayer(saveObject.savedSize);
			return layer;
		}
	}
	Ment.LeakyReluLayer = LeakyReluLayer;
	Ment.LeakyRelu = LeakyReluLayer;
	Ment.LRelu = LeakyReluLayer;
}
