{
	class BiasLayer {
		//this layer outputs the inputs with no changes
		constructor(size) {
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(size); //the inData
			this.outData = new Float32Array(size); //will be init when "connect" is called.
			this.b = new Float32Array(size);
			this.bs = new Float32Array(size);
			this.costs = new Float32Array(size); //costs for each neuron
			this.pl; //reference to previous layer
			this.trainIterations = 0;
			this.lr = 0;
		}

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			// try to connect to it
			if (this.inData.length == 0) {
				//if not already initialized
				this.inData = new Float32Array(layer.outSize());
				this.outData = new Float32Array(layer.outSize());
				this.costs = new Float32Array(layer.outSize());
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

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = this.inData[h] + this.b[h];
			}
		}

		backward(expected) {
			this.trainIterations++;
			let loss = 0;
			if (!expected) {
				if (this.nextLayer == undefined) {
					throw 'nothing to backpropagate!';
				}
				expected = [];
				for (var i = 0; i < this.outData.length; i++) {
					this.costs[i] = this.nextLayer.costs[i];
					this.bs[i] += this.nextLayer.costs[i];
					loss += this.costs[i];
				}
			} else {
				for (var j = 0; j < this.outData.length; j++) {
					let err = expected[j] - this.outData[j];
					this.costs[j] = err;
					this.bs[i] += err;

					loss += Math.pow(err, 2);
				}
			}
			return loss / this.inSize();
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
				if (key == 'pl' || key == 'inData' || key == 'outData' || key == 'costs' || key == 'nextLayer' || key == 'previousLayer') {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;

			return ret;
		}


		updateParams(optimizer) {
			if (this.trainIterations < 0) {
				return;
			}
			if (optimizer == 'SGD') {

					for (var j = 0; j < this.outSize(); j++) {
						this.bs[j] /= this.trainIterations;
						this.b[j] += this.bs[j] * this.lr;
						this.bs[j] = 0;
					}
				

				this.trainIterations = 0;
			}
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new BiasLayer(saveObject.savedSize);
			for (var i = 0; i < layer.b.length; i++) {
				layer.b[i] = saveObject.b[i];
			}
			return layer;
		}
	}

	Ment.BiasLayer = BiasLayer;
	Ment.Bias = BiasLayer;

}
