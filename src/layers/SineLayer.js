{
	class SineLayer extends Ment.ActivationBase {
		sine(z) {
			return Math.sin(z);
		}

		sinePrime(z) {
			return Math.cos(z);
		}

		constructor(size) {
			super(size);
		}

		forward(inData) {
			return super.forward(inData, this.sine);
		}

		backward(expected) {
			super.backward(expected, this.sinePrime);
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}
		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new SineLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.SineLayer = SineLayer;
	Ment.Sine = SineLayer;
	Ment.Sin = SineLayer;
}
