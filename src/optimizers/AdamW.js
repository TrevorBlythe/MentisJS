{
	class AdamW {
		constructor(maxEpochBeforeReset = 100, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0) {
			this.beta1 = beta1;
			this.beta2 = beta2;
			this.eps = eps;
			this.maxEpochBeforeReset = maxEpochBeforeReset;
			this.weightDecay = weightDecay;
			this.netObject; // will be set to the net object
			this.m = []; // first moment estimates
			this.v = []; // second moment estimates
			this.t = 0; // iteration counter
		}

		initialize() {
			// Access parameters and gradients from netObject
			const paramsAndGrads = this.netObject.getParamsAndGrads();

			// Initialize first and second moment estimates for each parameter
			for (let i = 0; i < paramsAndGrads.length / 2; i++) {
				const params = paramsAndGrads[i * 2];
				this.m[i] = new Float64Array(params.length).fill(0.0);
				this.v[i] = new Float64Array(params.length).fill(0.0);
			}
		}

		applyGradients(paramsAndGrads) {
			this.t += 1;
			this.weightDecay += this.eps;
			for (let i = 0; i < paramsAndGrads.length / 2; i++) {
				const params = paramsAndGrads[i * 2];
				const grads = paramsAndGrads[i * 2 + 1];
				this.netObject.smoothGradients(grads);
				for (let k = 0; k < params.length; k++) {
					// Update first moment estimate
					this.m[i][k] = this.beta1 * this.m[i][k] + (1 - this.beta1) * grads[k];

					// Update second moment estimate
					this.v[i][k] = this.beta2 * this.v[i][k] + (1 - this.beta2) * grads[k] * grads[k];

					// Correct bias for first and second moment estimates
					let correctedM = this.m[i][k] / (1 - Math.pow(this.beta1, this.t));
					let correctedV = this.v[i][k] / (1 - Math.pow(this.beta2, this.t));

					// Calculate update value with AdamW formula using learning rate from netObject
					let update = this.netObject.learningRate * correctedM;
					update /= Math.sqrt(correctedV) + this.eps;

					// Update weight parameter
					params[k] += update;
					params[k] -= this.weightDecay * -params[k];
				}
			}
			if (this.netObject.epoch % this.maxEpochBeforeReset == 0) {
				this.t = 0;
				this.weightDecay = 0;
				//fill m and v with zero
				for (let i = 0; i < this.m.length; i++) {
					this.m[i].fill(0.0);
					this.v[i].fill(0.0);
				}
			}
		}
	}
	Ment.AdamW = AdamW;
	Ment.AdamW = AdamW;
}
