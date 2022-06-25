{
	class SigmoidLayer extends Ment.ActivationBase {
		sigmoid(z) {
			return 1 / (1 + Math.exp(-z));
		}

		sigmoidPrime(z) {
			let ret = Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2);
			if (isNaN(ret)) {
				return 0;
			}
			return ret;
		}

		constructor(size) {
			super(size);
		}

		forward(inData) {
			super.forward(inData, this.sigmoid);
		}

		backward(expected) {
			return super.backward(expected, this.sigmoidPrime);
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new SigmoidLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.SigmoidLayer = SigmoidLayer;
	Ment.Sigmoid = SigmoidLayer;
	Ment.Sig = SigmoidLayer;
}
