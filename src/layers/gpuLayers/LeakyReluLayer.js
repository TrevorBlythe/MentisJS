{
	class LeakyReluLayerGPU extends Ment.ActivationBaseGPU {
		static leakySlope = 0.01;

		constructor(size) {
			super(size);
		}

		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = ${this.gpuInDataName}(${this.gpuInDataStartIndex} + i);

if(act < 0.0){
	act = act * ${LeakyReluLayerGPU.leakySlope};
}

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
				this.outSize(),
				`
float act = ${this.gpuErrorArrayName}(i);

if(${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) < 0.0){
	act = act * ${LeakyReluLayerGPU.leakySlope};
}

;

${this.gpuGradsArrayName}(i) := act;
				`
			);
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new LeakyReluLayerGPU(saveObject.savedSize);
			return layer;
		}
	}
	Ment.LeakyReluLayerGPU = LeakyReluLayerGPU;
	Ment.LeakyReluGPU = LeakyReluLayerGPU;
	Ment.LReluGPU = LeakyReluLayerGPU;
}
