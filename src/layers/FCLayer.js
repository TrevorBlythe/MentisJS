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
		based on the cost of the next layer (this.nextLayer.grads).

		REQUIRED FUNCTIONS:
			inSize()
			outSize()
			forward(inData::array)
			backward(ExpectedOutput::array) -- returns loss of the layer (Expected - outData)^2
			updateParams(optimizer::string) // optimizer examples: "SGD" "adam" "adagrad" "adadelta"
			save() -- returns json string of layer

		REQUIRED STATIC FUNCTIONS:
			load(json) -- returns a layer  EX: FCLayer.load(layer.save()); //makes a copy of "layer"
			
		REQUIRED FEILDS:
			inData:Float64Array
			outData:Float64Array
			grads:Float64Array -- the error value of each input neuron
			
			----Everything above is the bare minimum for your layer to work at all--------------
			
		NOT REQUIRED FUNCTIONS:
			inSizeDimensions() -- returns array of dimensions i.e. [28,28,3], 
			this function is used to print layers prettier
			outSizeDimensions()
			
		FEILDS SET BY NET OBJECT:
			nextLayer -- a reference to the layer after
			previousLayer -- a reference to preceding layer		
			you dont have to implement these, they will be set by the net object

		FUNCTIONS FOR EVOLOUTION BASED LEARNING TO WORK:
			mutate(mutationRate::float, mutationIntensity::float)
			only required if you want to "evolve" a network

		
			If you have all this your layer should work inside a real net!
			If you want it to work for gpu you need to implement the gpu version of the layer
		

	*/

	class FCLayer {
		constructor(inSize, outSize, useBias) {
			this.useBias = useBias == undefined ? true : useBias;
			this.ws = new Float64Array(inSize * outSize); //the weights sensitivities to error
			this.bs = new Float64Array(outSize); //the bias sensitivities to error
			this.nextLayer; //the connected layer
			this.inData = new Float64Array(inSize); //the inData
			this.outData = new Float64Array(outSize);
			this.w = new Float64Array(inSize * outSize); //this will store the weights
			this.b = new Float64Array(outSize); //this will store the biases (biases are for the outData (next layer))
			this.grads = new Float64Array(inSize); //grads for each neuron

			for (var j = 0; j < inSize; j++) {
				//----init random weights
				for (var h = 0; h < outSize; h++) {
					// this.w[h + j * outSize] = Math.sqrt(1.0 / inSize) * (Math.random() - 0.5) * 2;
					this.w[h + j * outSize] = (Math.random() - 0.5) * 2;
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
					weightGrads[j + i * outSize] += inData[i] * err[j];
					grads[i] += weights[j + i * outSize] * err[j];
				}
			}

			for (var j = 0; j < outSize; j++) {
				biasGrads[j] += err[j]; //bias grad so easy
			}
			//finish averaging the grads : required code
			// for (var i = 0; i < inSize; i++) {
			// 	grads[i] = grads[i] / outSize;
			// }
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
			let layer = new FCLayer(saveObject.savedInSize, saveObject.savedOutSize, saveObject.useBias);
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

	Ment.FCLayer = FCLayer;
	Ment.FC = FCLayer;
}
