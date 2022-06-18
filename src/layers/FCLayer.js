{
	/*
		Layer specification:

		Each layer needs an inData and outData array.
		These arrays should never be replaced because they are shared objects.
		This means never do this. "this.inData = [];" EXCEPT in the constructor.
		Replacing the inData or outData array will de-connect the layers.

		Each layer needs a "forward" and a "backward" function. The forward function should
		be able to take an input array as input or use the data in "this.inData." The same for the 
		backward function, it should take an array of "expected" values or it should backpropogate
		based on the cost of the next layer (this.nextLayer.costs).

		REQUIRED FUNCTIONS:
			inSize()
			outSize()
			forward(inData::array) --returns outData
			backward(inData::array) --returns loss of the layer
			updateParams(optimizer::string) // optimizer examples: "SGD" "adam" "adagrad" "adadelta"

		NOT REQUIRED FUNCTIONS:
			inSizeDimensions() -- returns array of dimensions i.e. [28,28,3]
			outSizeDimensions()
		
		REQUIRED PARAMETERS:
			inData
			outData
			lr -- aka learning rate
			costs
			nextLayer
			gpuEnabled

		FUNCTIONS FOR EVOLOUTION BASED LEARNING:
			mutate(mutationRate::float, mutationIntensity::float)

		FUNCTIONS FOR GPU SUPPORT:
			initGPU() -- this should at least make "this.gpuEnabled" true.

			You can code the rest yourself, swagnet uses gpujs and swag.gpu is a 
			global gpu object to use to prevent making multpile gpu objects. 
			Make it work however you want.

			If you have all this your layer should work inside a real net!

		

	*/

	class FCLayer {
		constructor(inSize, outSize) {
			if (!inSize) {
				throw 'Missing parameter in FC Constructor: (input size)';
			}

			if (!outSize) {
				throw 'Missing parameter in FC Constructor: (output size)';
			}

			this.lr = 0.01; //learning rate, this will be set by the Net object
			//dont set it in the constructor unless you really want to

			this.gpuEnabled = false;
			this.trainIterations = 0; //++'d whenever backwards is called;
			this.ws = new Float32Array(inSize * outSize); //the weights sensitivities to error
			this.bs = new Float32Array(outSize); //the bias sensitivities to error
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(inSize); //the inData
			this.outData = new Float32Array(outSize);
			this.w = new Float32Array(inSize * outSize); //this will store the weights
			this.b = new Float32Array(outSize); //this will store the biases (biases are for the outData (next layer))
			this.costs = new Float32Array(inSize); //costs for each neuron

			for (var j = 0; j < inSize; j++) {
				//----init random weights
				for (var h = 0; h < outSize; h++) {
					this.w[h + j * outSize] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
					this.ws[h + j * outSize] = 0;
				}
			} // ---- end init weights

			for (var j = 0; j < outSize; j++) {
				this.b[j] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				this.bs[j] = 0;
			} ///---------adding random biases
		}

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw 'INPUT SIZE WRONG ON FC LAYER:\nexpected size (' + this.inSize() + '), got: (' + inData.length + ')';
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = 0; //reset outData activation
				for (var j = 0; j < this.inSize(); j++) {
					this.outData[h] += this.inData[j] * this.w[h + j * this.outSize()]; // the dirty deed
				}
				this.outData[h] += this.b[h];
			}
		}

		backward(expected) {
			this.trainIterations++;
			let loss = 0;
			for (var i = 0; i < this.inSize(); i++) {
				//reset the costs
				this.costs[i] = 0;
			}

			if (!expected) {
				// -- sometimes the most effiecant way is the least elagant one...
				if (this.nextLayer == undefined) {
					throw 'error backproping on an unconnected layer with no expected parameter input';
				}
				for (var j = 0; j < this.outSize(); j++) {
					let err = this.nextLayer.costs[j];
					loss += Math.pow(err, 2);
					for (var i = 0; i < this.inSize(); i++) {
						//activation times error = change to the weight.
						this.ws[j + i * this.outSize()] += this.inData[i] * err * 2;
						this.costs[i] += this.w[j + i * this.outSize()] * err * 2;
					}
					//bias grad is real simple :)
					this.bs[j] += err;
				}
			} else {
				for (var j = 0; j < this.outSize(); j++) {
					let err = expected[j] - this.outData[j];
					loss += Math.pow(err, 2);
					for (var i = 0; i < this.inSize(); i++) {
						//activation times error = change to the weight.
						this.ws[j + i * this.outSize()] += this.inData[i] * err * 2;
						this.costs[i] += this.w[j + i * this.outSize()] * err * 2;
					}
					//bias grad is real simple :)
					this.bs[j] += err;
				}
			} // end of 'sometimes the most effeciant is the least elagant' ----

			//finish averaging the costs
			for (var i = 0; i < this.inSize(); i++) {
				this.costs[i] = this.costs[i] / this.outSize();
			}

			return loss / this.outSize();
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		updateParams(optimizer) {
			if (this.trainIterations < 0) {
				return;
			}
			if (optimizer == 'SGD') {
				for (var i = 0; i < this.ws.length; i++) {
					this.ws[i] /= this.trainIterations;
					this.w[i] += this.ws[i] * this.lr;
					this.ws[i] = 0;
				}

				for (var j = 0; j < this.outSize(); j++) {
					this.bs[j] /= this.trainIterations;
					this.b[j] += this.bs[j] * this.lr;
					this.bs[j] = 0;
				}

				this.trainIterations = 0;
			}
		}

		mutate(mutationRate, mutationIntensity) {
			for (var i = 0; i < this.w.length; i++) {
				if (Math.random() < mutationRate) {
					this.w[i] += Math.random() * mutationIntensity * (Math.random() > 0.5 ? -1 : 1);
				}
			}
			for (var i = 0; i < this.b.length; i++) {
				if (Math.random() < mutationRate) {
					this.b[i] += Math.random() * mutationIntensity * (Math.random() > 0.5 ? -1 : 1);
				}
			}
		}

		//----------------
		// GPU specific stuff below! Beware.
		//----------------

		// initGPU(){
		// 	this.forwardGPUKernel = swag.gpu.createKernel(function(inData, weights, biases, outDataLength, inDataLength) {
		// 		var act = 0;
		// 		for (var j = 0; j < inDataLength; j++) {
		// 					act +=
		// 						inData[j] *
		// 						weights[this.thread.x + j *	outDataLength];// the dirty deed
		// 		}
		// 		act += biases[this.thread.x];
		// 		return act
		// 	}).setOutput([this.outData.length]);
		// }

		//----------------
		// End of GPU specific stuff. Take a breather.
		//----------------
	}

	swag.FCLayer = FCLayer;
	swag.FC = FCLayer;
}
