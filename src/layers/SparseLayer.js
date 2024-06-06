{
	/*
		This is like the FC Layer except it uses a sparse matrix
		YOU CAN CONSTRUCT IN TWO WAYS
		SparseLayer(InSize, OutSize, useBias)
		SparseLayer(FCLayer, Density) // higher density means more weights (0.5 is 50% of the weights are used)
		//weights get culled off
	*/

	class SparseLayer {
		constructor(inSize, outSize, useBias) {
			console.log("Dont use SparseLayer i havent fully implemented it yet and it 100% does not work.");
			if (inSize.constructor.name == "FCLayer") {
				let layer = inSize;
				let density = outSize;
				this.useBias = layer.useBias;
				this.nextLayer = layer.nextLayer;
				this.inData = new Float64Array(layer.inData);
				this.outData = new Float64Array(layer.outData);
				this.w = new Float64Array(layer.w);
				this.weightIndexes = new Float64Array(layer.outSize());
				//fill weightIndexes with 1,2,3,4.....layer.outSize()
				for (var i = 0; i < layer.outSize(); i++) {
					this.weightIndexes[i] = i;
				}
				this.b = new Float64Array(layer.b);
				this.ws = new Float64Array(layer.ws);
				this.bs = new Float64Array(layer.bs);
				this.grads = new Float64Array(layer.grads);
				this.cullWeights(density);
				return;
			}
			this.useBias = useBias == undefined ? true : useBias;
			this.nextLayer; //the connected layer
			this.inData = new Float64Array(inSize); //the inData
			this.outData = new Float64Array(outSize);

			this.w = new Float64Array(inSize * outSize); //this will store the weights, in no particular order
			//but the index of the weights does correspond the in the input data index it is connected to
			this.weightIndexes = new Float64Array(inSize * outSize); //this will store the index of the outData the weight connects to
			this.b = new Float64Array(outSize); //this will store the biases (biases are for the outData (next layer))
			this.ws = new Float64Array(inSize * outSize); //the weights sensitivities to error
			this.bs = new Float64Array(outSize); //the bias sensitivities to error
			this.grads = new Float64Array(inSize); //grads for each neuron

			for (var j = 0; j < inSize; j++) {
				//----init random weights
				for (var h = 0; h < outSize; h++) {
					this.w[h + j * outSize] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
					this.ws[h + j * outSize] = 0;
				}
			} // ---- end init weights

			for (var j = 0; j < outSize; j++) {
				if (this.useBias == false) {
					this.b[j] = 0;
				} else {
					this.b[j] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
				this.bs[j] = 0;
			} ///---------adding random biases
		}

		//this is the forward function
		//it takes an input array and feeds it through the layer
		//it will also use the inData if no input is given
		forward(input) {
			// console.log(inData);
			if (input) {
				if (input.length != this.inSize()) {
					throw inputError(this, input);
				}
				//Fun fact: the fastest way to copy an array in javascript is to use a For Loop
				for (var i = 0; i < input.length; i++) {
					this.inData[i] = input[i];
				}
			}
			// console.log(this.inData);
			const outData = this.outData;
			const inData = this.inData;
			const biases = this.b;
			const outSize = this.outSize();
			const weights = this.w;
			const inSize = this.inSize();
			for (var h = 0; h < outSize; h++) {
				outData[h] = biases[h]; //reset outData activation
				for (var j = 0; j < inSize; j++) {
					outData[h] += inData[j] * weights[h + j * outSize]; // the dirty deed
				}
			}
		}

		//explain what this layer is actualy doing
		//this is the backward function
		//it takes an array of expected values and calculates the error
		backward(err) {
			if (!err) {
				err = this.nextLayer.grads;
			}

			const inData = this.inData;
			const outSize = this.outSize();
			const weights = this.w;
			const inSize = this.inSize();
			const grads = this.grads;
			const weightGrads = this.ws;
			const biasGrads = this.bs;

			for (var i = 0; i < inSize; i++) {
				grads[i] = 0;
				for (var j = 0; j < outSize; j++) {
					//activation times error = change to the weight.
					weightGrads[j + i * outSize] += inData[i] * err[j] * 2;
					grads[i] += weights[j + i * outSize] * err[j] * 2;
				}
			}

			for (var j = 0; j < outSize; j++) {
				biasGrads[j] += err[j]; //bias grad so easy
			}
			//finish averaging the grads : required code
			for (var i = 0; i < inSize; i++) {
				grads[i] = grads[i] / outSize;
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		getParamsAndGrads(forUpdate = true) {
			if (this.useBias) {
				return [this.w, this.ws, this.b, this.bs];
			} else {
				return [this.w, this.ws];
			}
		}

		mutate(mutationRate, mutationIntensity) {
			for (var i = 0; i < this.w.length; i++) {
				if (Math.random() < mutationRate) {
					this.w[i] += Math.random() * mutationIntensity * (Math.random() > 0.5 ? -1 : 1);
				}
			}
			if (this.useBias) {
				for (var i = 0; i < this.b.length; i++) {
					if (Math.random() < mutationRate) {
						this.b[i] += Math.random() * mutationIntensity * (Math.random() > 0.5 ? -1 : 1);
					}
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
					key == "ws" ||
					key == "bs" ||
					key == "inData" ||
					key == "netObject" ||
					key == "outData" ||
					key == "grads" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
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
			let layer = new SparseLayer(saveObject.savedInSize, saveObject.savedOutSize, saveObject.useBias);
			for (var i = 0; i < layer.w.length; i++) {
				layer.w[i] = saveObject.w[i];
			}
			if (layer.useBias) {
				for (var i = 0; i < layer.b.length; i++) {
					layer.b[i] = saveObject.b[i];
				}
			}
			return layer;
		}
	}

	Ment.SparseLayer = SparseLayer;
	Ment.Sparse = SparseLayer;
}
