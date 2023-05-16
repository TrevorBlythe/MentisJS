{
	/*
        This is the basic white girl of optimizers. Doesn't even do anything to the gradient.
        batch size is already built in to the network so we dont even have to do that here
    */
	class SGD {
		constructor() {
			this.netObject; //will be set to the net object
		} //END OF CONSTRUCTOR

		//this function gets called after this.netObject is set by the network.
		initialize() {}

		applyGradients(paramsAndGrads) {
			for (var j = 0; j < paramsAndGrads.length; j += 2) {
				let params = paramsAndGrads[j];
				let grads = paramsAndGrads[j + 1];
				for (var k = 0; k < params.length; k++) {
					params[k] += (grads[k] / this.netObject.iteration) * this.netObject.learningRate;
					grads[k] = 0;
				}
			}
		}
	} //END OF SGD CLASS DECLARATION

	Ment.SGD = SGD;
}
