{
	class TanhLayer extends Ment.ActivationBase {
		tanh(z) {
			return Math.tanh(z);
		}

		tanhPrime(z) {
			return 1 - Math.pow(Math.tanh(z), 2);
		}

		constructor(size) {
			super(size);
		}

		forward(inData) {
			super.forward(inData, this.tanh);
		}

		backward(expected) {
			return super.backward(expected, this.tanhPrime);
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new TanhLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.TanhLayer = TanhLayer;
	Ment.Tanh = TanhLayer;
}
