{
	class DataBlockLayer {
		//this layer outputs the inputs with no changes
		constructor(size) {
			this.nextLayer; //the connected layer
			this.inData = new Float64Array(size); //the inData
			this.outData = new Float64Array(size); //will be init when "connect" is called.
			this.grads = new Float64Array(size); //grads for each neuron
			this.pl; //reference to previous layer
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
				this.grads = new Float64Array(layer.outSize());
			}
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

			this.outData.fill(0);
		}

		backward(err) {
			this.grads.fill(0); //best layer
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "netObject" ||
					key == "pl" ||
					key == "outData" ||
					key == "grads" ||
					key == "nextLayer" ||
					key == "previousLayer"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new DataBlockLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.DataBlockLayer = DataBlockLayer;
	Ment.Block = DataBlockLayer;
	Ment.Zeroes = DataBlockLayer;
	Ment.Wall = DataBlockLayer;
}
