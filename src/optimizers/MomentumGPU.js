{
	/*
        Gives the gradients "momentum" down the slope. Nothing special to it but
		it is rumored that it increased training speed a lot! I think it might be 
		a good idea to use this instead of SGD

		Super excited about it. 
    */
	class MomentumGPU {
		constructor(momentum = 0.9) {
			this.netObject; //will be set to the net object
			this.lastGrads = [];
			this.momentum = momentum; // a constant which represents how strong the gradients "momentum" will be
		} //END OF CONSTRUCTOR

		initialize() {
			//we need to get "this.lastGrads" list the right size. It needs to be as long as there are gradients in this network.
			let pag = this.netObject.getParamsAndGrads();
			for (var j = 0; j < pag.length; j += 2) {
				let grads = pag[j + 1];
				Ment.webMonkeys.set(this.lastGrads[this.lastGrads.push(Ment.makeid(8)) - 1], Ment.webMonkeys.getLen(grads));
			}
		}
		applyGradients(paramsAndGrads) {
			for (var j = 0; j < paramsAndGrads.length; j += 2) {
				let params = paramsAndGrads[j];
				let grads = paramsAndGrads[j + 1];
				// for (var k = 0; k < params.length; k++) {
				// 	params[k] += (grads[k] / this.netObject.iteration) * this.netObject.learningRate + this.momentum * this.lastGrads[j][k];
				// 	this.lastGrads[gradCounter] =
				// 		(grads[k] / this.netObject.iteration) * this.netObject.learningRate + this.momentum * this.lastGrads[j][k];
				// 	grads[k] = 0;
				// 	gradCounter++;
				// }
				Ment.webMonkeys.work(
					Ment.webMonkeys.getLen(params),
					`
float act = ${params}(i) + (${grads}(i) / ${parseFloat(this.netObject.batchSize).toFixed(1)}) * ${parseFloat(
						this.netObject.learningRate
					).toFixed(8)} + ${parseFloat(this.momentum).toFixed(8)} * ${this.lastGrads[j / 2]}(i);

;

${params}(i) := act;

`
				);

				Ment.webMonkeys.work(
					Ment.webMonkeys.getLen(params),
					`
float act = (${grads}(i) / ${parseFloat(this.netObject.batchSize).toFixed(8)}) * ${parseFloat(
						this.netObject.learningRate
					).toFixed(8)} + ${parseFloat(this.momentum).toFixed(8)} * ${this.lastGrads[j / 2]}(i);

;

${this.lastGrads[j / 2]}(i) := act;

`
				);

				Ment.webMonkeys.fill(grads, 0);
			}
		}
	} //END OF Momentum CLASS DECLARATION

	Ment.MomentumGPU = MomentumGPU;
}
