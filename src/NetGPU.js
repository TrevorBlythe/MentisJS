var Ment = Ment || {};

{
	class NetGPU {
		constructor(layers, optimizer = "SGDGPU") {
			if (!Ment.webMonkeys) {
				console.log("GPU wont work until you include the altered webmonkeys code");
			}
			this.layers = layers || [];
			this.batchSize = 5; //every 'batchSize' iterations, update the weights by calling 'updateParams' (resets iteration to 0)
			this.epoch = 0; //goes up every 'updateParams' call
			this.iteration = 0; //goes up every 'backwards' call, goes back to 0 when 'updateParams' call
			this.learningRate = 0.01;
			this.gpuEnabled = false; //turns on when gpu mode is enabled
			this.gpuLastLayerExpectedArrayName = Ment.makeid(8); //generates random 8 character string
			this.gpuFirstLayerInput = Ment.makeid(8);

			//Putting this here to keep code from the OG days alive.
			if (optimizer == "SGDGPU") {
				this.optimizer = new Ment.SGDGPU();
			} else if (typeof optimizer == "string") {
				this.optimizer = new Ment[optimizer]();
			} else {
				this.optimizer = optimizer;
			}
			this.optimizer.netObject = this; //give it access to the net object so it can get the learning rate and stuff.
			this.optimizer.initialize();
			this.connectLayers();
		} //END OF CONSTRUCTOR

		get inData() {
			return this.layers[0].inData;
		}

		set inData(input) {
			if (input.length == this.layers[0].inSize()) {
				Ment.webMonkeys.set(this.layers[0].gpuInDataName, input);
			} else {
				throw inputError(this.layers[0], input);
			}
		}

		get outData() {
			return [...this.layers[this.layers.length - 1].outData];
		}

		connectLayers() {
			//'connect' means make the outData array the same object as the inData of the next layer.
			//this is so we dont have to copy over the arrays values when forwarding/backwarding
			//Also we set the 'nextLayer' and 'previousLayer' attributes to each layer accordingly
			//We set these properties BEFORE the indata/outdata connecting so layers can initialize array sizes.(they can sometimes be infered)
			//this is because some layers rely on surrounding layers to initialize, (like Sigmoid, you dont need to put in the size)
			//where you woudnt want to have to input the size as a parameter, so it automatically initializes if
			// you dont.)

			let activationIds = Array(this.layers.length + 1); // an id for every set of activations
			let activationErrorIds = Array(this.layers.length + 1); // an id for every set of activations errors
			let activationsId = Ment.makeid(8);
			for (var i = 0; i < this.layers.length + 1; i++) {
				activationIds[i] = activationsId; //Ment.makeid(8);
				activationErrorIds[i] = Ment.makeid(8);
				//theres a very small chance to layers get the same id for something
				//this would mess everything up but the chance is so small and all u gotta do is reload the code
			}

			for (var i = 1; i < this.layers.length; i++) {
				this.layers[i].previousLayer = this.layers[i - 1];
			}

			for (var i = this.layers.length - 1; i >= 0; i--) {
				this.layers[i].nextLayer = this.layers[i + 1];
			}

			for (var i = 0; i < this.layers.length - 1; i++) {
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
				// this.layers[i].initGPUActivations(
				// 	activationIds[i],
				// 	activationIds[i + 1],
				// 	activationErrorIds[i],
				// 	activationErrorIds[i + 1]
				// );
				this.layers[i].gpuInDataName = activationIds[i];
				this.layers[i].gpuOutDataName = activationIds[i + 1];
				this.layers[i].gpuCostsArrayName = activationErrorIds[i];
				Ment.webMonkeys.set(activationErrorIds[i], this.layers[i].inSize());

				this.layers[i].gpuErrorArrayName = activationErrorIds[i + 1];
				Ment.webMonkeys.set(activationErrorIds[i + 1], this.layers[i].outSize());
			}
			let runningTotal = 0;
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].gpuInDataStartIndex = runningTotal;
				runningTotal += this.layers[i].inSize();
				this.layers[i].gpuOutDataStartIndex = runningTotal;
				// runningTotal += this.layers[i].outSize();
			}
			runningTotal += this.layers[this.layers.length - 1].outSize();

			Ment.webMonkeys.set(activationsId, runningTotal);
			this.layers[this.layers.length - 1].gpuInDataName = activationIds[this.layers.length - 1];
			this.layers[this.layers.length - 1].gpuOutDataName = activationIds[this.layers.length];
			this.layers[this.layers.length - 1].gpuCostsArrayName = activationErrorIds[this.layers.length - 1];
			Ment.webMonkeys.set(activationErrorIds[this.layers.length - 1], this.layers[this.layers.length - 1].inSize());
			this.layers[this.layers.length - 1].gpuErrorArrayName = activationErrorIds[this.layers.length];
			Ment.webMonkeys.set(activationErrorIds[this.layers.length], this.layers[this.layers.length - 1].outSize());
		}

		forward(data) {
			if (data != undefined && !Array.isArray(data) && !data.constructor.name.endsWith("Array")) {
				if (typeof data == "number") {
					data = [data];
				} else {
					throw "ONLY INPUT ARRAYS INTO FORWARDS FUNCTION! you inputted: " + data.constructor.name;
				}
			}
			if (data) {
				Ment.webMonkeys.set(this.gpuFirstLayerInput, data);
				Ment.webMonkeys.work(
					this.layers[0].inSize(),
					`
float act = ${this.gpuFirstLayerInput}(i);

${this.layers[0].gpuInDataName}(i) := act;
			`
				);
			}
			// this.layers[0].forward(data);
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].forward();
			}

			//getting the outdata from the gpu is to expensive to do it everytime
			//if the user wants the outdata they will have to call for it
			//net.outData
			// return this.layers[this.layers.length - 1].outData;
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

			//convert any non gpu layers to gpu layers
			let jsonObj = JSON.parse(json);
			let layers = [];
			for (var i = 0; i < jsonObj.layerAmount; i++) {
				if (!jsonObj["layer" + i].type.endsWith("GPU")) {
					jsonObj["layer" + i].type += "GPU";
				}
				let layer = Ment[jsonObj["layer" + i].type].load(jsonObj["layer" + i].layerData);
				layers.push(layer);
			}
			if (!jsonObj.optimizer.endsWith("GPU")) {
				jsonObj.optimizer += "GPU";
			}
			let ret = new Ment.NetGPU(layers, jsonObj.optimizer);
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

		backward(expected, calcLoss = false, backPropErrorInstead = false) {
			//returns the loss of the last layer only cuz im actually such a lazy lard
			let loss = 0;
			let lastLayer = this.layers[this.layers.length - 1];
			if (expected) {
				if (expected.length != this.layers[this.layers.length - 1].outSize()) {
					throw `Error: The array your trying to train your AI on, is the wrong size. ${expected.length}:${this.layers[
						this.layers.length - 1
					].outSize()}`;
				}

				//first calculate the error (expected - actual) and pass it to
				//the last layer.
				if (!backPropErrorInstead) {
					// err = new Float32Array(lastlayer.outSize());
					// for (var i = 0; i < lastlayer.outSize(); i++) {
					// 	//for every outAct in last layer
					// 	err[i] = expected[i] - lastlayer.outData[i];
					// }

					Ment.webMonkeys.set(this.gpuLastLayerExpectedArrayName, expected);

					Ment.webMonkeys.work(
						lastLayer.outSize(),
						`
${lastLayer.gpuErrorArrayName}(i) := ${this.gpuLastLayerExpectedArrayName}(i) - ${lastLayer.gpuOutDataName}(i + ${lastLayer.gpuOutDataStartIndex});

`
					);
				} else {
					Ment.webMonkeys.set(lastLayer.gpuErrorArrayName, expected);
				}
				if (calcLoss) {
					let err = Ment.webMonkeys.get(lastLayer.gpuErrorArrayName);
					for (var i = 0; i < err.length; i++) {
						loss += Math.pow(err[i], 2); //always positive
					}
					loss /= expected.length; //average it
				}
			}

			lastLayer.backward(); //no need to pass in the error, the above code sets it all up
			for (var i = this.layers.length - 2; i >= 0; i--) {
				//it automatically grabs the calcualted error from the layer ahead.
				//Would work the same as if you did backward(nextlayer.costs)
				//Costs represent the error of the indata neurons
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
					for (var j = 0; j < Ment.webMonkeys.getLen(pag); j += 2) {
						let grads = pag[j + 1];
						Ment.webMonkeys.fill(grads, 0);
					}
				}
			}
		}

		updateParams() {
			this.epoch++;
			let pag = this.getParamsAndGrads();
			this.optimizer.applyGradients(pag);
		}
	} //END OF NET CLASS DECLARATION

	Ment.NetGPU = NetGPU;
}
