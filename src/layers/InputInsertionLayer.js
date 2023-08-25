{
	class InputInsertionLayer {
		//this layer allows you to add input to the data flow of the model at any
		//place in the layers of the model. By calling "setInput" before each
		//forward call, you can have two inputs. This can be useful for situations
		//where you wanna input an image AND maybe also a latent vector somewhere
		//in the middle of the model. Or if you wanna add some kind of time vector
		//All this layer does is take the current input and adds it to the activations
		//of the last layer.
		constructor(size, inputSize) {
			if (size == undefined || inputSize == undefined) {
				throw "You need to define size and input size for InputInsertionLayer";
			}
			this.nextLayer; //the connected layer
			this.inData = new Float64Array(size); //the inData
			this.outData = new Float64Array(size + inputSize); //will be init when "connect" is called.
			this.costs = new Float64Array(size); //costs for each neuron
			this.currentInput = new Float64Array(inputSize);
			this.pl; //reference to previous layer
		}

		setInput(input) {
			for (var i = 0; i < input.length; i++) {
				this.currentInput[i] = input[i];
			}
		}

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			// try to connect to it
			// if (this.inData.length == 0) {
			// 	//if not already initialized
			// 	this.inData = new Float64Array(layer.outSize());
			// 	this.outData = new Float64Array(layer.outSize());
			// 	this.costs = new Float64Array(layer.outSize());
			// }
			this.pl = layer;
		}
		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw Ment.inputError(this, inData);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.inSize(); h++) {
				this.outData[h] = this.inData[h];
			}

			for (var h = 0; h < this.currentInput.length; h++) {
				this.outData[this.inSize() + h] = this.currentInput[h];
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			//the currentInput is not something trainable so we just ignore
			//the error it causes and backprop everything else
			for (var j = 0; j < this.inData.length; j++) {
				this.costs[j] = err[j];
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		save() {
			this.savedSize = this.inSize();
			this.savedInputSize = this.currentInput.length;
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "pl" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "currentInput"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;
			delete this.savedInputSize;

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new InputInsertionLayer(saveObject.savedSize, saveObject.savedInputSize);
			return layer;
		}
	}

	Ment.InputInsertionLayer = InputInsertionLayer;
	Ment.InputInsertion = InputInsertionLayer;
	Ment.ActivationInsertion = InputInsertionLayer;
}
