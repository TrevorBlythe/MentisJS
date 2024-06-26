var Ment = Ment || {};
{
	class Net {
		constructor(layers, optimizer = "SGD") {
			this.layers = layers || [];
			this.batchSize = 5; //every 'batchSize' iterations, update the weights by calling 'updateParams' (resets iteration to 0)
			this.epoch = 0; //goes up every 'updateParams' call
			this.iteration = 0; //goes up every 'backwards' call, goes back to 0 when 'updateParams' call
			this.learningRate = 0.01;
			//Putting this here to keep code from the OG days alive.
			this.connectLayers();
			if (optimizer == "SGD") {
				this.optimizer = new Ment.SGD();
			} else if (typeof optimizer == "string") {
				this.optimizer = new Ment[optimizer]();
			} else {
				this.optimizer = optimizer;
			}
			this.optimizer.netObject = this; //give it access to the net object so it can get the learning rate and stuff.
			this.optimizer.initialize();
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

		get inSize() {
			return this.layers[0].inSize();
		}

		get outData() {
			return this.layers[this.layers.length - 1].outData;
		}

		get outSize() {
			return this.layers[this.layers.length - 1].outSize();
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

			for (let i = 0; i < this.layers.length; i++) {
				this.layers[i].netObject = this;
			}

			for (var i = 1; i < this.layers.length; i++) {
				this.layers[i].previousLayer = this.layers[i - 1];
			}

			for (var i = this.layers.length - 1; i >= 0; i--) {
				this.layers[i].nextLayer = this.layers[i + 1];
			}

			for (var i = 0; i < this.layers.length - 1; i++) {
				if (this.layers[i].outSize() != this.layers[i + 1].inSize()) {
					// A very complex error message that simply states that the layers are not connected properly and gives there dimensions. And it the layer doesnt have any
					//dimensions, it will check the layer before it if it has the same size.
					throw `Failure connecting ${
						this.layers[i].constructor.name + (this.layers[i].id ? `(${this.layers[i].id})` : null)
					} layer with ${this.layers[i + 1].constructor.name},${this.layers[i].constructor.name} output size: ${this.layers[
						i
					].outSize()}${
						this.layers[i].outSizeDimensions
							? " (" + this.layers[i].outSizeDimensions() + ")"
							: this.layers[i - 1]
							? this.layers[i - 1].outSizeDimensions && this.layers[i - 1].outSize() == this.layers[i].inSize()
								? "(" + this.layers[i - 1].outSizeDimensions() + ")"
								: ""
							: ""
					}, ${this.layers[i + 1].constructor.name} input size: ${this.layers[i + 1].inSize()}${
						this.layers[i + 1].inSizeDimensions ? " (" + this.layers[i + 1].inSizeDimensions() + ")" : ""
					}`;
				}
				this.layers[i + 1].inData = this.layers[i].outData;
			}
		}

		forward(data) {
			if (data != undefined && !Array.isArray(data) && !data.constructor.name.endsWith("Array")) {
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

		mutate(rate, intensity) {
			for (var i = 0; i < this.layers.length; i++) {
				if (this.layers[i].mutate) {
					this.layers[i].mutate(rate, intensity);
				}
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
			saveObject.optimizer = this.optimizer.constructor.name;
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
				let sus = jsonObj["layer" + i].type;
				if (jsonObj["layer" + i].type.endsWith("GPU")) {
					sus = jsonObj["layer" + i].type.slice(0, -3);
				}
				let layer = Ment[sus].load(jsonObj["layer" + i].layerData);
				layers.push(layer);
			}
			let sus = jsonObj.optimizer;
			if (sus.endsWith("GPU")) {
				sus = sus.slice(0, -3);
			}
			let ret = new Ment.Net(layers, sus);
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

		backward(expected, calcLoss = true, backPropErrorInstead = false) {
			//returns the loss of the last layer only cuz im actually such a lazy lard
			let loss = 0;
			let err = undefined;
			let lastlayer = this.layers[this.layers.length - 1];
			if (expected) {
				if (expected.length != this.layers[this.layers.length - 1].outData.length) {
					throw `Error: The array your trying to train your AI on, is the wrong size. ${expected.length}:${
						this.layers[this.layers.length - 1].outData.length
					}`;
				}
				//first calculate the error (expected - actual) and pass it to
				//the last layer.
				if (!backPropErrorInstead) {
					err = new Float64Array(lastlayer.outData.length);
					for (var i = 0; i < lastlayer.outData.length; i++) {
						//for every outAct in last layer
						err[i] = expected[i] - lastlayer.outData[i];
					}
				} else {
					err = expected;
				}
				if (calcLoss) {
					for (var i = 0; i < err.length; i++) {
						loss += Math.pow(err[i], 2); //always positive
					}
					loss /= expected.length; //average it
				}
			}
			lastlayer.backward(err);
			for (var i = this.layers.length - 2; i >= 0; i--) {
				//it automatically grabs the calcualted error from the layer ahead.
				//Would work the same as if you did backward(nextlayer.grads)
				//Grads represent the error of the indata neurons
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

		clearGrads() {
			//clears the gradients
			for (var i = this.layers.length - 1; i >= 0; i--) {
				if (this.layers[i].getParamsAndGrads) {
					let pag = this.layers[i].getParamsAndGrads();
					for (var j = 0; j < pag.length; j += 2) {
						let grads = pag[j + 1];
						for (var k = 0; k < grads.length; k++) {
							grads[k] = 0;
						}
					}
				}
			}
		}

		//adjusting this multiplier might help
		smoothGradients(gradients, multiplier = 0.7) {
			// 			  """
			//   Smoothly adjusts gradients in-place based on their distance from the threshold,
			//   with a target range of 0 (at threshold) to 2 (far from threshold).
			//   Args:
			//     gradients: A list of numbers representing the gradients (modified in-place).
			//     multiplier: A factor to multiply the mean absolute value by (default: 5).
			//   """
			// Calculate mean absolute value
			// let meanAbs = gradients.reduce((sum, value) => sum + Math.abs(value), 0) / gradients.length;
			//cannot use reduce function
			let sum = 0;
			for (let i = 0; i < gradients.length; i++) {
				sum += Math.abs(gradients[i]);
			}
			const meanAbs = sum / gradients.length;

			// console.log(meanAbs);
			const threshold = meanAbs * multiplier;
			// Smooth adjustment function (avoids creating a new array)
			for (let i = 0; i < gradients.length; i++) {
				const gradient = gradients[i];
				const distanceFromThreshold = Math.abs(gradient) - meanAbs;
				const adjustmentFactor = multiplier / (1 + Math.exp(-distanceFromThreshold)); //sigmoid function
				// console.log(adjustmentFactor);
				gradients[i] = adjustmentFactor * gradient;
			}
		}

		updateParams() {
			this.epoch++;
			let pag = this.getParamsAndGrads();
			this.optimizer.applyGradients(pag);
		}
	} //END OF NET CLASS DECLARATION

	Ment.Net = Net;
}
