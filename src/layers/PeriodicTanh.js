{
	class PeriodicTanhLayer extends Ment.ActivationBase {
		// I came up with this one myself lol
		Tanh(z) {
			return Math.tanh(Math.sin(z) * 4);
		}

		TanhPrime(x) {
			//2 / (Math.exp(x) + Math.exp(-x))
			let y = Math.cos(2 * x);
			let ret;
			if (y > 0) {
				ret = Math.cos(x) * Math.cos(2 * x) * Math.cos(2 * x);
			} else {
				ret = -x * 0.001;
			}

			return ret;
		}

		constructor(size) {
			super(size);
		}

		forward(inData) {
			super.forward(inData, this.Tanh);
		}

		backward(expected) {
			return super.backward(expected, this.TanhPrime);
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new PeriodicTanhLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.PeriodicTanhLayer = PeriodicTanhLayer;
	Ment.PeriodicTanh = PeriodicTanhLayer;
	Ment.PeriodicTanh = PeriodicTanhLayer;
	Ment.PTanh = PeriodicTanhLayer;
}
