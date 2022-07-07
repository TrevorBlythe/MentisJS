var Ment = Ment || {};
{
	class Net {
		constructor(layers, optimizer) {
			this.layers = layers || [];
			this.batchSize = 19;
			this.epoch = 0;
			this.lr = 0.01;
			this.optimizer = optimizer || 'SGD';

			this.connectLayers();
		} //END OF CONSTRUCTOR

		set learningRate(lr) {
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].lr = lr;
			}
			this.lr = lr;
		}

		get learningRate() {
			return this.lr;
		}

		get inData() {
			return this.layers[0].inData;
		}

		set inData(arr) {
			if (arr.length == this.layers[0].inSize()) {
				this.layers[0].inData = arr;
			} else {
				throw 'cant set in data because its the wrong size';
			}
		}

		get outData() {
			return [...this.layers[this.layers.length - 1].outData];
		}

		set outData(arr) {
			if (arr.length == this.layers[this.layers.length - 1].outSize()) {
				this.layers[0].outData = arr;
			} else {
				throw 'cant set out data because its the wrong size';
			}
		}

		connectLayers() {
			//by 'connect' we mean make the outData the same object as the inData of adjacent arrays
			//this is so we dont have to copy over the arrays values when forwarding/backwarding
			//Also we set the 'nextLayer' and 'previousLayer' attributes to each layer accordingly
			for (var i = 0; i < this.layers.length; i++) {
				if (i < this.layers.length - 1) {
					this.layers[i + 1].previousLayer = this.layers[i];
					this.layers[i].nextLayer = this.layers[i + 1];

					if (this.layers[i].outSize() != this.layers[i + 1].inSize()) {
						throw `Failure connecting ${this.layers[i].constructor.name} layer with ${this.layers[i + 1].constructor.name},${
							this.layers[i].constructor.name
						} output size: ${this.layers[i].outSize()}${this.layers[i].outSizeDimensions ? ' (' + this.layers[i].outSizeDimensions() + ')' : ''}, ${
							this.layers[i + 1].constructor.name
						} input size: ${this.layers[i + 1].inSize()}${this.layers[i + 1].inSizeDimensions ? ' (' + this.layers[i + 1].inSizeDimensions() + ')' : ''}`;
					}
					this.layers[i + 1].inData = this.layers[i].outData;
				}
			}

		}

		forward(data) {
			if (!Array.isArray(data)) {
				if (typeof data == 'number') {
					data = [data];
				} else {
					throw 'INPUT ARRAYS INTO FORWARDS FUNCTION ONLY!';
				}
			}
			this.layers[0].forward(data);
			for (var i = 1; i < this.layers.length; i++) {
				this.layers[i].forward();
			}
			return new Float32Array(this.layers[this.layers.length - 1].outData); //return inData of the last layer (copied btw so you can change it if you want)
		}

		enableGPU() {
			console.log('gpu hasnt been implemented yet so nothing will happen');
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].gpuEnabled = true;
				if (this.layers[i].initGPU) {
					this.layers[i].initGPU();
				}
			}
		}

		mutate(rate, intensity) {
			for (var i = 0; i < this.layers.length; i++) {
				if (this.layers[i].mutate) {
					this.layers[i].mutate(rate, intensity);
				}
			}
		}

		disableGPU() {
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].gpuEnabled = false;
			}
		}

		save(saveToFile, filename) {
			//this can be made better.
			//this method returns a string of json used to get the model back

			if (saveToFile) {
				if (Ment.isBrowser()) {
					var a = document.createElement('a');
					var file = new Blob([this.save()]);
					a.href = URL.createObjectURL(file);
					a.download = 'model.json';
					a.click();
				} else {
					var fs = require('fs');
					fs.writeFileSync(filename, this.save());
				}
			}

			let saveObject = {};

			saveObject.layerAmount = this.layers.length;
			saveObject.optimizer = this.optimizer;
			saveObject.lr = this.lr;
			saveObject.batchSize = this.batchSize;
			for (var i = 0; i < this.layers.length; i++) {
				let layer = this.layers[i];
				let layerSaveObject = {};
				layerSaveObject.type = layer.constructor.name;
				saveObject['layer' + i] = layerSaveObject;
				if (!layer.save) {
					throw `Layer ${i} (${layer.constructor.name}) in your network doesnt have a save() function so your model cant be saved`;
				}
				layerSaveObject.layerData = layer.save();
			}

			return JSON.stringify(saveObject);
		} //end of save method

		static load(json) {
			let jsonObj = JSON.parse(json);
			let layers = [];
			for (var i = 0; i < jsonObj.layerAmount; i++) {
				let layer = Ment[jsonObj['layer' + i].type].load(jsonObj['layer' + i].layerData);
				layers.push(layer);
			}
			let ret = new Ment.Net(layers, jsonObj.optimizer);
			ret.learningRate = jsonObj.lr;
			ret.batchSize = jsonObj.batchSize;
			return ret;
		} //end of load method

		copy(net) {
			/* This method copies the weights and bes of one network into its own weights and bes  */
		} //end of copy method

		train(input, expectedOut) {
			this.forward(input);

			let loss = this.backward(expectedOut);
			//done backpropping
			return loss;
		}

		backward(expected) {
			let loss = this.layers[this.layers.length - 1].backward(expected);
			for (var i = this.layers.length - 2; i >= 0; i--) {
				this.layers[i].backward();
			}
			this.epoch++;
			if (this.epoch % this.batchSize == 0) {
				for (var i = this.layers.length - 1; i >= 0; i--) {
					if (this.layers[i].updateParams) this.layers[i].updateParams(this.optimizer);
				}
			}
			return loss;
		}
	} //END OF NET CLASS DECLARATION

	Ment.Net = Net;
}
