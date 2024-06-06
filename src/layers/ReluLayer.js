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

		backward(err) {
			if (!err) {
				err = this.nextLayer.grads;
			}

			for (var j = 0; j < this.outSize(); j++) {
				if (this.outData[j] > 0) {
					this.grads[j] = err[j];
				} else {
					this.grads[j] = 0;
				}
			}
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
