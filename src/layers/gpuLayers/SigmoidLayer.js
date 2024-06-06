{
	class SigmoidLayerGPU extends Ment.ActivationBaseGPU {
		constructor(size) {
			super(size);
		}

		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = 1.0 / (1.0 + exp(-${this.gpuInDataName}(i + ${this.gpuInDataStartIndex})));

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
float zee = exp(-${this.gpuInDataName}(i + ${this.gpuInDataStartIndex}));
float act = ${this.gpuErrorArrayName}(i) * (zee / ((1.0 + zee) * (1.0 + zee)));

;

${this.gpuGradsArrayName}(i) := act;
				`
			);
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new SigmoidLayerGPU(saveObject.savedSize);
			return layer;
		}
	}

	Ment.SigmoidLayerGPU = SigmoidLayerGPU;
	Ment.SigmoidGPU = SigmoidLayerGPU;
	Ment.SigGPU = SigmoidLayerGPU;
}
