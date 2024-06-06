{
	class InputInsertionLayerGPU {
		//this layer allows you to add input to the data flow of the model at any
		//place in the layers of the model. By calling "setInput" before each
		//forward call, you can have two inputs. This can be useful for situations
		//where you wanna input an image AND maybe also a latent vector somewhere
		//in the middle of the model. Or if you wanna add some kind of time vector
		//All this layer does is take the current input and adds it to the activations
		//of the last layer.
		constructor(size, inputSize) {
			if (size == undefined || inputSize == undefined) {
				throw "You need to define size and input size for InputInsertionLayerGPU";
			}
			this.nextLayer; //the connected layer
			this.currentInputName = Ment.makeid(8);
			this.currentInputLength = inputSize;
			Ment.webMonkeys.set(this.currentInputName, inputSize);
			this.pl; //reference to previous layer
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;

			this.inSize = () => {
				return size;
			};

			this.outSize = () => {
				return inputSize + size;
			};
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		setInput(input) {
			if (input.length == this.outSize() - this.inSize()) {
				Ment.webMonkeys.set(this.currentInputName, input);
			}
		}

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			// try to connect to it
			// if (this.inData.length == 0) {
			// 	//if not already initialized
			// 	this.inData = new Float32Array(layer.outSize());
			// 	this.outData = new Float32Array(layer.outSize());
			// 	this.grads = new Float32Array(layer.outSize());
			// }
			this.pl = layer;
		}
		forward() {
			Ment.webMonkeys.work(
				this.inSize(),
				`
float act = ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
			);

			Ment.webMonkeys.work(
				this.currentInputLength,
				`
float act = ${this.currentInputName}(i);

;

${this.gpuOutDataName}(i + ${this.inSize()} + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			Ment.webMonkeys.work(
				this.inSize(),
				`
float act = ${this.gpuErrorArrayName}(i);

;

${this.gpuGradsArrayName}(i) := act;
				`
			);
		}

		save() {
			this.savedSize = this.inSize();
			this.savedInputSize = Ment.webMonkeys.getLen(this.currentInputName);
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "netObject" ||
					key == "pl" ||
					key == "outData" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuGradsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "grads" ||
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
			let layer = new InputInsertionLayerGPU(saveObject.savedSize, saveObject.savedInputSize);
			return layer;
		}
	}

	Ment.InputInsertionLayerGPU = InputInsertionLayerGPU;
	Ment.InputInsertionGPU = InputInsertionLayerGPU;
	Ment.ActivationInsertionGPU = InputInsertionLayerGPU;
}
