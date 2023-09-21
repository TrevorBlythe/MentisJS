{
	class LayerNormalization {
		constructor(size) {
			//For some reason these layers are supposed to be rescaled and shifted so they
			//have weights and biases i guess
			this.gs = new Float64Array(size); //the gammas sensitivities to error
			this.bs = new Float64Array(size); //the bias (beta) sensitivities to error
			this.nextLayer; //the connected layer
			this.inData = new Float64Array(size); //the inData
			this.outData = new Float64Array(size); //will be init when "connect" is called.
			this.g = new Float64Array(size); //this will store the weights
			this.b = new Float64Array(size); //this will store the biases (biases are for the outData (next layer))
			this.costs = new Float64Array(size); //costs for each neuron
			this.outDataBeforeRescaleAndShift = new Float64Array(size);
			this.standard_deviation = 1; //this will hold the last computed standard deviations
			for (var j = 0; j < size; j++) {
				this.g[j] = 1;
				this.b[j] = 0;
				this.gs[j] = 0;
				this.bs[j] = 0;
			} // ---- end init params
		}

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			// try to connect to it
			if (this.inData.length == 0) {
				//if not already initialized
				this.inData = new Float64Array(layer.outSize());
				this.outData = new Float64Array(layer.outSize());
				this.costs = new Float64Array(layer.outSize());
				this.valMinusMeanSquared = new Float64Array(layer.outSize());
				this.outDataBeforeRescaleAndShift = new Float64Array(layer.outSize());
				this.g = new Float64Array(layer.outSize());
				this.b = new Float64Array(layer.outSize());
				this.gs = new Float64Array(layer.outSize());
				this.bs = new Float64Array(layer.outSize());

				for (var j = 0; j < layer.outSize(); j++) {
					this.g[j] = 1;
					this.b[j] = 0;
					this.gs[j] = 0;
					this.bs[j] = 0;
				} // ---- end init params
			}
			this.pl = layer;
		}

		forward(input) {
			if (input) {
				if (input.length != this.inSize()) {
					throw inputError(this, input);
				}
				//Fun fact: the fastest way to copy an array in javascript is to use a For Loop
				for (var i = 0; i < input.length; i++) {
					this.inData[i] = input[i];
				}
			}

			const outData = this.outData;
			const inData = this.inData;
			const biases = this.b;
			const outSize = this.outSize();
			const gammas = this.g;
			const inSize = this.inSize();
			const outDataBeforeRescaleAndShift = this.outDataBeforeRescaleAndShift;
			const mean = inData.reduce((a, b) => a + b) / inSize;
			this.standard_deviation =
				Math.sqrt(inData.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / inSize) + Number.EPSILON;

			const std_dev = this.standard_deviation;
			//rescale and shift part
			for (var h = 0; h < outSize; h++) {
				outDataBeforeRescaleAndShift[h] = (inData[h] - mean) / std_dev;
				outData[h] = outDataBeforeRescaleAndShift[h] * gammas[h] + biases[h];
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			const outSize = this.outSize();
			const gammas = this.g;
			const costs = this.costs;
			const weightGrads = this.gs;
			const biasGrads = this.bs;

			//mean and standard deviation aint trainable

			for (var j = 0; j < outSize; j++) {
				weightGrads[j] += this.outDataBeforeRescaleAndShift[j] * err[j];
				biasGrads[j] += err[j];
				costs[j] = (gammas[j] * err[j]) / this.standard_deviation;
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		getParamsAndGrads(forUpdate = true) {
			return [this.g, this.gs, this.b, this.bs];
		}

		mutate(mutationRate, mutationIntensity) {
			for (var i = 0; i < this.g.length; i++) {
				if (Math.random() < mutationRate) {
					this.g[i] += Math.random() * mutationIntensity * (Math.random() > 0.5 ? -1 : 1);
				}
			}
			for (var i = 0; i < this.b.length; i++) {
				if (Math.random() < mutationRate) {
					this.b[i] += Math.random() * mutationIntensity * (Math.random() > 0.5 ? -1 : 1);
				}
			}
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			this.savedInSize = this.inSize();
			this.savedOutSize = this.outSize();
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "gs" ||
					key == "bs" ||
					key == "inData" ||
					key == "outData" ||
					key == "outDataBeforeRescaleAndShift" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "standard_deviation" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == (this.useBias ? null : "b")
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;
			delete this.savedOutSize;

			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new LayerNormalization(saveObject.savedInSize);
			for (var i = 0; i < layer.g.length; i++) {
				layer.g[i] = saveObject.g[i];
			}
			if (layer.useBias) {
				for (var i = 0; i < layer.b.length; i++) {
					layer.b[i] = saveObject.b[i];
				}
			}
			return layer;
		}
	}

	Ment.LayerNormalization = LayerNormalization;
	Ment.LayerNorm = LayerNormalization;
}
