{
	/*
       SGD except it works for GPU AI instead of cpu ai
    */
	class SGDGPU {
		constructor() {
			this.netObject; //will be set to the net object
		} //END OF CONSTRUCTOR

		//this function gets called after this.netObject is set by the network.
		initialize() {}

		applyGradients(paramsAndGrads) {
			for (var j = 0; j < paramsAndGrads.length; j += 2) {
				let params = paramsAndGrads[j];
				let grads = paramsAndGrads[j + 1];
				// params[k] += (grads[k] / this.netObject.iteration) * this.netObject.learningRate;
				// grads[k] = 0;
				let len = Ment.webMonkeys.getLen(params);
				Ment.webMonkeys.work(
					len,
					`
float act = ${params}(i) + (${grads}(i) / ${parseFloat(this.netObject.batchSize).toFixed(8)}) * ${parseFloat(
						this.netObject.learningRate
					).toFixed(8)};

;

${params}(i) := act;

`
				);
				Ment.webMonkeys.fill(grads, 0);
			}
		}
	} //END OF SGD CLASS DECLARATION

	Ment.SGDGPU = SGDGPU;
}
