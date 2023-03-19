var Ment = Ment || {};
{
	class Net {
		constructor(layers, optimizer) {
			this.layers = layers || [];
			this.batchSize = 5; //every 'batchSize' iterations, update the weights by calling 'updateParams' (resets iteration to 0)
			this.epoch = 0; //goes up every 'updateParams' call
			this.iteration = 0; //goes up every 'backwards' call, goes back to 0 when 'updateParams' call
			this.learningRate = 0.01;
			this.optimizer = optimizer || "SGD"; //I have not made any other optimizer's yet

			this.connectLayers();
		} //END OF CONSTRUCTOR

		get inData() {
			return this.layers[0].inData;
		}

		set inData(arr) {
			if (arr.length == this.layers[0].inSize()) {
				this.layers[0].inData = arr;
			} else {
				throw "cant set in data because its the wrong size";
			}
		}

		get outData() {
			return [...this.layers[this.layers.length - 1].outData];
		}

		set outData(arr) {
			if (arr.length == this.layers[this.layers.length - 1].outSize()) {
				this.layers[this.layers.length - 1].outData = arr;
			} else {
				throw "cant set out data because its the wrong size";
			}
		}

		connectLayers() {
			//'connect' means make the outData array the same object as the inData of the next layer.
			//this is so we dont have to copy over the arrays values when forwarding/backwarding
			//Also we set the 'nextLayer' and 'previousLayer' attributes to each layer accordingly
			//We set these properties BEFORE the indata/outdata connecting so layers can initialize array sizes.(they can sometimes be infered)
			//this is because some layers rely on surrounding layers to initialize, (like Sigmoid, you dont need to put in the size)
			//where you woudnt want to have to input the size as a parameter, so it automatically initializes if
			// you dont.)

			//this function is always safe to call. some data may get reset though.

			for (var i = 1; i < this.layers.length; i++) {
				this.layers[i].previousLayer = this.layers[i - 1];
			}

			for (var i = this.layers.length - 1; i >= 0; i--) {
				this.layers[i].nextLayer = this.layers[i + 1];
			}

			for (var i = 0; i < this.layers.length; i++) {
				if (i < this.layers.length - 1) {
					if (this.layers[i].outSize() != this.layers[i + 1].inSize()) {
						throw `Failure connecting ${
							this.layers[i].constructor.name + (this.layers[i].id ? `(${this.layers[i].id})` : null)
						} layer with ${this.layers[i + 1].constructor.name},${this.layers[i].constructor.name} output size: ${this.layers[
							i
						].outSize()}${this.layers[i].outSizeDimensions ? " (" + this.layers[i].outSizeDimensions() + ")" : ""}, ${
							this.layers[i + 1].constructor.name
						} input size: ${this.layers[i + 1].inSize()}${
							this.layers[i + 1].inSizeDimensions ? " (" + this.layers[i + 1].inSizeDimensions() + ")" : ""
						}`;
					}
					this.layers[i + 1].inData = this.layers[i].outData;
				}
			}
		}

		forward(data) {
			if (!Array.isArray(data) && data.constructor.name != "Float32Array") {
				if (typeof data == "number") {
					data = [data];
				} else {
					throw "ONLY INPUT ARRAYS INTO FORWARDS FUNCTION! you inputted: " + data.constructor.name;
				}
			}
			this.layers[0].forward(data);
			for (var i = 1; i < this.layers.length; i++) {
				this.layers[i].forward();
			}
			return this.layers[this.layers.length - 1].outData;
		}

		enableGPU() {
			console.log("gpu hasnt been implemented yet so nothing will happen");
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
					var a = document.createElement("a");
					var file = new Blob([this.save()]);
					a.href = URL.createObjectURL(file);
					a.download = "model.json";
					a.click();
				} else {
					var fs = require("fs");
					fs.writeFileSync(filename, this.save());
				}
			}

			let saveObject = {};

			saveObject.layerAmount = this.layers.length;
			saveObject.optimizer = this.optimizer;
			saveObject.learningRate = this.learningRate;
			saveObject.batchSize = this.batchSize;
			for (var i = 0; i < this.layers.length; i++) {
				let layer = this.layers[i];
				let layerSaveObject = {};
				layerSaveObject.type = layer.constructor.name;
				saveObject["layer" + i] = layerSaveObject;
				if (!layer.save) {
					throw `Layer ${i} (${layer.constructor.name}) in your network doesnt have a save() function so your model cant be saved`;
				}
				layerSaveObject.layerData = layer.save();
			}

			return JSON.stringify(saveObject);
		} //end of save method

		static load(json) {
			//this function takes json from the "save" function
			let jsonObj = JSON.parse(json);
			let layers = [];
			for (var i = 0; i < jsonObj.layerAmount; i++) {
				let layer = Ment[jsonObj["layer" + i].type].load(jsonObj["layer" + i].layerData);
				layers.push(layer);
			}
			let ret = new Ment.Net(layers, jsonObj.optimizer);
			ret.learningRate = jsonObj.learningRate;
			ret.batchSize = jsonObj.batchSize;
			return ret;
		} //end of load method

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
			this.iteration++;
			if (this.iteration % this.batchSize == 0) {
				this.updateParams();
				this.iteration = 0;
			}
			return loss;
		}

		getParamsAndGrads() {
			//The word "params" could also be seen as "weights" here.
			let ret = [];
			for (var i = 0; i < this.layers.length; i++) {
				if (this.layers[i].getParamsAndGrads) {
					let pag = this.layers[i].getParamsAndGrads();
					for (var j = 0; j < pag.length; j += 2) {
						let params = pag[j];
						let grads = pag[j + 1];
						ret.push(params);
						ret.push(grads);
					}
				}
			}
			return ret;
		}

		updateParams() {
			this.epoch++;
			for (var i = this.layers.length - 1; i >= 0; i--) {
				if (this.layers[i].getParamsAndGrads) {
					let pag = this.layers[i].getParamsAndGrads();
					for (var j = 0; j < pag.length; j += 2) {
						let params = pag[j];
						let grads = pag[j + 1];
						for (var k = 0; k < params.length; k++) {
							params[k] += (grads[k] / this.iteration) * this.learningRate; //only does SGD rn
							// params[k] += Ment.protectNaN((grads[k] / this.batchSize) * this.learningRate); //safe version
							grads[k] = 0;
						}
					}
				}
			}
		}
	} //END OF NET CLASS DECLARATION

	Ment.Net = Net;
}
