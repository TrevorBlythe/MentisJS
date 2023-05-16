{
	/*
        Gives the gradients "momentum" down the slope. Nothing special to it but
		it is rumored that it increased training speed a lot! I think it might be 
		a good idea to use this instead of SGD

		Super excited about it. 
    */
	class Momentum {
		constructor(momentum = 0.9) {
			this.netObject; //will be set to the net object
			this.lastGrads = [];
			this.momentum = momentum; // a constant which represents how strong the gradients "momentum" will be
		} //END OF CONSTRUCTOR

		initialize() {
			//we need to get "this.lastGrads" list the right size. It needs to be as long as there are gradients in this network.
			let pag = this.netObject.getParamsAndGrads();
			var gradCounter = 0;
			for (var j = 0; j < pag.length; j += 2) {
				let grads = pag[j + 1];
				gradCounter += grads.length;
			}
			this.lastGrads = new Float32Array(gradCounter);
			this.lastGrads.fill(0);
		}
		applyGradients(paramsAndGrads) {
			var gradCounter = 0;
			for (var j = 0; j < paramsAndGrads.length; j += 2) {
				let params = paramsAndGrads[j];
				let grads = paramsAndGrads[j + 1];
				for (var k = 0; k < params.length; k++) {
					params[k] +=
						(grads[k] / this.netObject.iteration) * this.netObject.learningRate + this.momentum * this.lastGrads[gradCounter];
					this.lastGrads[gradCounter] =
						(grads[k] / this.netObject.iteration) * this.netObject.learningRate + this.momentum * this.lastGrads[gradCounter];
					grads[k] = 0;
					gradCounter++;
				}
			}
		}
	} //END OF Momentum CLASS DECLARATION

	Ment.Momentum = Momentum;
}
