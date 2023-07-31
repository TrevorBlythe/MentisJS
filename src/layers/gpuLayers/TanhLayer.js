{
	class TanhLayerGPU extends Ment.ActivationBaseGPU {
		tanh(z) {
			return Math.tanh(z);
		}

		tanhPrime(z) {
			return 1 - Math.pow(Math.tanh(z), 2);
		}

		constructor(size) {
			super(size);
		}

		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float e = 2.71828;
float act = 2.0 / (1.0 + pow(e,-2.0 * ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex}))) - 1.0;

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			Ment.webMonkeys.work(
				this.inSize(),
				`
float e = 2.71828;
float zee = 2.0 / (1.0 + pow(e,-2.0 * ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex}))) - 1.0;
float act = 1.0 - (zee * zee);
act = act * ${this.gpuErrorArrayName}(i);
;

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new TanhGPU(saveObject.savedSize);
			return layer;
		}
	}

	Ment.TanhLayerGPU = TanhLayerGPU;
	Ment.TanhGPU = TanhLayerGPU;
}
