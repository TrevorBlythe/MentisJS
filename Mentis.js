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
			if (optimizer == "SGD") {
				this.optimizer = new Ment.SGD();
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
				this.layers[i + 1].inData = this.layers[i].outData;
			}
		}

		forward(data) {
			if (data != undefined && !Array.isArray(data) && data.constructor.name != "Float32Array") {
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
					err = new Float32Array(lastlayer.outData.length);
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
					for (var j = 0; j < pag.length; j += 2) {
						let grads = pag[j + 1];
						for (var k = 0; k < grads.length; k++) {
							grads[k] = 0;
						}
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

	Ment.Net = Net;
}

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
			if (data != undefined && !Array.isArray(data) && data.constructor.name != "Float32Array") {
				if (typeof data == "number") {
					data = [data];
				} else {
					throw "ONLY INPUT ARRAYS INTO FORWARDS FUNCTION! you inputted: " + data.constructor.name;
				}

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

var Ment = Ment || {};
{
	class Population {
		constructor(netLayers, popSize, elitism, mutationRate, mutationIntensity) {
			this.networks = [];
			this.popSize = popSize;
			this.highScore = 0;
			this.mutationRate = mutationRate || 0.3;
			this.mutationIntensity = mutationIntensity || 0.6;
			this.elitism = elitism; //how many of the best players stay after a culling
			let baseNet = new Ment.Net(netLayers);
			baseNet.score = 0;
			for (let i = 0; i < popSize - 1; i++) {
				let net = Ment.Net.load(baseNet.save());
				net.score = 0;
				this.networks.push(net);
			}
			this.networks.push(baseNet);
		}

		cullAndBreed() {
			//keeps the strongest players.
			//brutally kills the weakest

			//but first we have to sort them. good to bad
			this.networks.sort((a, b) => (a.score > b.score ? -1 : 1));
			this.highScore = this.networks[0].score < this.highScore ? this.highScore : this.networks[0].score;
			//next we put the best ones in an array for later
			let newPopulation = [];
			for (let i = 0; i < this.elitism; i++) {
				this.networks[i].score = 0;
				newPopulation.push(this.networks[i]);
				//this actually only pushes the references btw.
				//which is fine
			}
			//next we have to generate enough offspring to fill the rest of the pop
			//higher scores have a higher chance of making offspring
			//"making offspring" just means a mutated net is made from it.
			for (let i = 0; i < this.popSize - this.elitism; i++) {
				for (let j = 0; j < this.popSize; j++) {
					if (Math.random() < 0.3) {
						//doing this with a sorted population should randomly choose
						//the best ones to breed
						let t = Ment.Net.load(this.networks[i].save());
						t.score = 0; //init score
						t.mutate(0.6, this.mutationRate);
						newPopulation.push(t);
						j = this.popSize + 5; //this is just to end the for loop
					}
					if (j == this.popSize - 1) {
						j = 0; //incase you just get super unlucky we have a fail safe.
					}
				}
			}

			// for (var i = 0; i < newPopulation.length; i++) {
			// 	newPopulation[i].score = 0;
			// }
			this.networks = newPopulation;
		}
	}

	Ment.Population = Population;
}

var Ment = Ment || {};
{
	class Rnn extends Ment.Net {
		/*
			EXPLANATION

			Everytime you forward something through this it will take a "snapshot"
			of all of the activations and store them. Ive called this a "state." (2 dim array savedStates[layer index][activation])

			Once your done forwarding and you begin backpropping in reverse order,
			the network will go back in time through its saved snapshots and will perform backpropagation
			as if it was still its old self. (look up backprop through time.)  The state will then be popped off.

			calling "backward()" without parameter will allow you to backprop the error from 
			the n+1th state to the nth state.

			Hidden state magic happens in the reccurentReceiver and reccurrentEmitter layers.

		*/
		constructor(layers, optimizer) {
			super(layers, optimizer);
			this.savedStates = []; //fills with arrays of arrays containing in/out Datas for the layers. need to save them
			this.trainingMode = true; //when this is on, it will save snapshots of the network for backprop through time.
			//to do backprop over time
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
				this.layers[0].outData = arr;
			} else {
				throw "cant set out data because its the wrong size";
			}
		}

		setState(ind) {
			if (ind == undefined) {
				ind = this.savedStates.length - 1;
			}
			let state = this.savedStates[ind];

			for (var i = 0; i < this.layers.length; i++) {
				let layer = this.layers[i];
				layer.inData = state[i];
				if (i > 0) {
					this.layers[i - 1].outData = layer.inData;
				}
			}
			this.layers[this.layers.length - 1].outData = state[state.length - 1];

			//special cases
			for (var i = 0; i < this.layers.length; i++) {
				if (this.layers[i].constructor.name == "ResReceiverLayer") {
					this.layers[i].inDataFromEmitter = this.layers[i].emitter.outData;
				}
			}
		}

		appendCurrentState() {
			//stores all current activations in a list formatted like this [[layer1 in activations], [layer2 in activations], .... [layer nth out activations]]
			//then it appends that array to 'savedStates'

			//could be optimized by just copying instead of straight up replacing the arrays? (both slow options)
			let state = this.savedStates[this.savedStates.push([]) - 1];
			for (var i = 0; i < this.layers.length; i++) {
				state.push(new Float32Array(this.layers[i].inData));
			}
			state.push(new Float32Array(this.layers[this.layers.length - 1].outData));
		}

		forward(data) {
			if (data == undefined) {
				//todo you dont need to make a whole new array here it should get copied by the layers forward function
				data = new Float32Array(this.layers[this.layers.length - 1].outData);
			}
			let ret = super.forward(data);
			//save the state
			if (this.trainingMode) {
				this.appendCurrentState();
			}
			return ret;
		}

		resetRecurrentData() {
			for (var i = 0; i < this.layers.length; i++) {
				let layer = this.layers[i];
				if (layer.constructor.name == "RecEmitterLayer") {
					layer.savedOutData.fill(0);
				}

				if (layer.constructor.name == "RecReceiverLayer") {
					layer.savedCostsForEmitter.fill(0);
				}
			}
		}

		train(input, expectedOut) {
			console.log("If your gonna use this method you might as well use a normal network");
			this.forward(input);

			let loss = this.backward(expectedOut);
			//done backpropping
			return loss;
		}

		backward(expected, calcLoss = true, backPropErrorInstead = false) {
			//expected represents error if backProperrorInstead == true

			//in rnns we might not have an expected array.. in this case..
			//backprop the error (first layers costs) from the "next iteration" (n+1).
			//this will only work if the indata and outdata of the network are the same size.
			//This means you must specify a Final output at least, which makes sense.
			//you could have a network with different in/out sizes but you would need to specify
			//the expected input and output at each step. - Trevor
			if (expected == undefined) {
				if (this.layers[0].inSize() != this.layers[this.layers.length - 1].outSize()) {
					throw "Size Mismatch in RNN. In data and out data are different dimensions.";
				}
				expected = new Float32Array(this.layers[0].inSize()); //maybe could just set it as reference instead of copy???
				for (var i = 0; i < this.layers[0].inSize(); i++) {
					expected[i] = this.layers[0].costs[i]; //from now on "expected" will represent backprop gradient or error.
				}
				backPropErrorInstead = true;
			}
			this.setState(); //this basically goes back in time by 1 step, search up backpropagation through time or something.
			this.savedStates.pop(); //we wont need this saved state anymore
			let loss = super.backward(expected, calcLoss, backPropErrorInstead);
			return loss;
		}

		static load(json) {
			let jsonObj = JSON.parse(json);
			let layers = [];
			for (var i = 0; i < jsonObj.layerAmount; i++) {
				let layer = Ment[jsonObj["layer" + i].type].load(jsonObj["layer" + i].layerData);
				layers.push(layer);
			}
			let ret = new Ment.Rnn(layers, jsonObj.optimizer);
			ret.learningRate = jsonObj.learningRate;
			ret.batchSize = jsonObj.batchSize;
			return ret;
		} //end of load method
	} //END OF Rnn CLASS DECLARATION

	Ment.Rnn = Rnn;
}

var Ment = Ment || {};
{
	class RnnGPU extends Ment.NetGPU {
		constructor(layers, optimizer) {
			super(layers, optimizer);
			this.savedStates = []; //fills with strings representing the activations
			this.currentState = -1; //starts at neg 1 dont change this
			this.trainingMode = true; //when this is on, it will save snapshots of the network for backprop through time.
		} //END OF CONSTRUCTOR

		setState(ind) {
			if (ind == undefined) {
				ind = this.currentState;
			}
			if (ind == -1) {
				return; //the state of having no state :()
			}
			let state = this.savedStates[ind];

			for (var i = 0; i < this.layers.length; i++) {
				let layer = this.layers[i];
				layer.gpuInDataName = state;
				if (i > 0) {
					this.layers[i - 1].gpuOutDataName = state;
				}
			}
			this.layers[this.layers.length - 1].gpuOutDataName = state;
		}

		pushNewState() {
			let newStateID = Ment.makeid(8);
			this.savedStates.push(newStateID);
			Ment.webMonkeys.set(newStateID, Ment.webMonkeys.getLen(this.layers[0].gpuInDataName));
		}

		forward(data) {
			if (this.trainingMode) {
				if (this.currentState == this.savedStates.length - 1) {
					this.pushNewState();
				}
				this.currentState++;
				this.setState(this.currentState);
			}
			if (data == undefined) {
				//todo you dont need to make a whole new array here it should get copied by the layers forward function
				// data = new Float32Array(this.layers[this.layers.length - 1].outData);
				Ment.webMonkeys.work(
					this.layers[0].inSize(),
					`
${this.layers[0].gpuInDataName}(i + ${this.layers[0].gpuInDataStartIndex}) := ${
						this.layers[this.layers.length - 1].gpuOutDataName
					}(i + ${this.layers[this.layers.length - 1].gpuOutDataStartIndex});
				`
				);
			}
			let ret = super.forward(data);
			//save the state

			return ret;
		}

		resetRecurrentData() {
			throw "todo";
			// for (var i = 0; i < this.layers.length; i++) {
			// 	let layer = this.layers[i];
			// 	if (layer.constructor.name == "RecEmitterLayer") {
			// 		layer.savedOutData.fill(0);
			// 	}
			// 	if (layer.constructor.name == "RecReceiverLayer") {
			// 		layer.savedCostsForEmitter.fill(0);
			// 	}
			// }
		}

		train(input, expectedOut) {
			console.log("If your gonna use this method you might as well use a normal network");
			this.forward(input);

			let loss = this.backward(expectedOut);
			//done backpropping
			return loss;
		}

		backward(expected, calcLoss = false, backPropErrorInstead = false) {
			//expected represents error if backProperrorInstead == true

			//in rnns we might not have an expected array.. in this case..
			//backprop the error (first layers costs) from the "next iteration" (n+1).
			//this will only work if the indata and outdata of the network are the same size.
			//This means you must specify a Final output at least, which makes sense.
			//you could have a network with different in/out sizes but you would need to specify
			//the expected input and output at each step. - Trevor
			if (expected == undefined) {
				if (this.layers[0].inSize() != this.layers[this.layers.length - 1].outSize()) {
					throw "Size Mismatch in RNN. In data and out data are different dimensions and/or you havent provided an input array";
				}
				// expected = new Float32Array(this.layers[0].inSize()); //maybe could just set it as reference instead of copy???
				// for (var i = 0; i < this.layers[0].inSize(); i++) {
				// 	expected[i] = this.layers[0].costs[i]; //from now on "expected" will represent backprop gradient or error.
				// }
				// backPropErrorInstead = true;
				Ment.webMonkeys.work(
					this.layers[0].inSize(),
					`
${this.layers[this.layers.length - 1].gpuErrorArrayName}(i) := ${this.layers[0].gpuCostsArrayName}(i);
				`
				);
			}
			let loss = super.backward(expected, calcLoss, backPropErrorInstead);
			this.currentState--;
			this.setState(this.currentState); //this basically goes back in time by 1 step, search up backpropagation through time or something.
			return loss;
		}

		static load(json) {
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
			let ret = new Ment.RnnGPU(layers, jsonObj.optimizer);
			ret.learningRate = jsonObj.learningRate;
			ret.batchSize = jsonObj.batchSize;
			return ret;
		} //end of load method
	} //END OF Rnn CLASS DECLARATION

	Ment.RnnGPU = RnnGPU;
}

var Ment = Ment || {};
{
	var return_v = false;
	var v_val = 0.0;

	var protectNaN = function (num) {
		if (num == Infinity) {
			return 1;
		} else if (num == -Infinity) {
			return -1;
		} else if (isNaN(num)) {
			return 0;
		} else {
			return num;
		}
	};

	function makeid(length) {
		//stolen code lol
		let result = "";
		const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
		const charactersLength = characters.length;
		let counter = 0;
		while (counter < length) {
			result += characters.charAt(Math.floor(Math.random() * charactersLength));
			counter += 1;
		}
		return result;
	}

	var printNet = function (net) {
		let text = "LAYERS:\n";
		for (var i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];

			text += `${i} ${net.layers[i].constructor.name} (${
				layer.inSizeDimensions ? "," + layer.inSizeDimensions() : layer.inSize()
			},${layer.outSizeDimensions ? "," + layer.outSizeDimensions() : layer.outSize()})${layer.id ? ` id:${layer.id}` : ""}\n`;
		}
		text += "STATS:\n";
		Object.keys(net).forEach(function (key, index) {
			if (key != "layers") {
				var value = net[key];
				text += `	${key}:${value}\n`;
			}
		});
		console.log(text);
	};

	var inputError = function (layer, arr) {
		let ret = `INPUT SIZE WRONG ON ${layer.constructor.name + (layer.id ? `(${layer.id})` : null)}: `;
		ret += `expected size (${layer.inSizeDimensions ? "," + layer.inSizeDimensions() : layer.inSize()}), got (${arr.length})`;
		return ret;
	};

	var gaussRandom = function () {
		if (return_v) {
			return_v = false;
			return v_val;
		}
		var u = 2 * Math.random() - 1;
		var v = 2 * Math.random() - 1;
		var r = u * u + v * v;
		if (r == 0 || r > 1) return gaussRandom();
		var c = Math.sqrt((-2 * Math.log(r)) / r);
		v_val = v * c; // cache this
		return_v = true;
		return u * c;
	};

	var isBrowser = () => !(typeof window === "undefined");
	//wrap everything in a namespace to not pollute global

	let clamp = function (num, min, max) {
		min = min | 0;
		max = max | 1;
		return Math.max(Math.min(num, max), min);
	};

	let lin = function (num) {
		return num;
	};

	let bounce = function (num, minmax) {
		minmax = minmax | 3;
		if (Math.abs(num) > minmax) {
			var t = (minmax * num) / Math.abs(num);
			num = t - (num - t);
			return bounce(num, minmax);
		}
		return num;
	};

	let getLoss = function (results, expected) {
		var loss = 0;
		for (var i = 0; i < results.length; i++) {
			loss += Math.pow(results[i] - expected[i], 2);
		}
		loss = loss / results.length;
		return loss;
	};

	let polluteGlobal = function () {
		for (const [key, value] of Object.entries(Ment)) {
			if (isBrowser()) {
				window[key] = value;
			} else {
				global[key] = value; //node
			}
		}
	};

	Ment.seed = 5;
	Ment.seededRandom = function (min, max) {
		max = max || 1;
		min = min || 0;

		Ment.seed = (Ment.seed * 9301 + 49297) % 233280;
		var rnd = Ment.seed / 233280;

		return min + rnd * (max - min);
	};

	Ment.seededRandomSP = function (min, max, seed) {
		//seeded random with seed as param
		max = max || 1;
		min = min || 0;

		Ment.seed = (Ment.seed * 9301 + 49297) % 233280;
		var rnd = Ment.seed / 233280;

		return min + rnd * (max - min);
	};

	let renderBox = function (options) {
		const net = options.net;
		const ctx = options.ctx;
		const x = options.x;
		const y = options.y;
		const scale = options.scale | 20;
		const spread = options.spread | 3;
		let background = options.background;
		if (background == undefined) {
			background = "white";
		}
		if (background == false) {
			background = "rgba(0,0,0,0)";
		}
		let maxSize = 1;
		for (var i = 0; i < net.layers.length; i++) {
			if (net.layers[i].inSize() > maxSize) {
				maxSize = net.layers[i].inSize();
			} //end of if
			if (net.layers[i].outSize() > maxSize) {
				maxSize = net.layers[i].outSize();
			} //end of if
		}

		ctx.fillStyle = background;

		ctx.fillRect(x, y, net.layers.length * (spread + scale) + scale, 7 * scale);
		for (var i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];
			let layerSize = layer.inSize() > layer.outSize() ? layer.inSize() : layer.outSize();
			layerSize = layerSize / maxSize; //is zero to one. one if its the max size, 0 if its zero size
			let layerLeftSize = layer.inSize() / maxSize; // is zero to one, same as above
			let layerRightSize = layer.outSize() / maxSize; // same as above

			Ment.seed = layer.constructor.name.charCodeAt(0) * 5; //------ make seed from layer name
			for (var n = 1; n < layer.constructor.name.length; n++) {
				Ment.seed += layer.constructor.name.charCodeAt(n);
			} //----------End of make random seed from layer name
			ctx.fillStyle = `rgb(${Ment.seededRandom(150, 255)},${Ment.seededRandom(100, 255)},${Ment.seededRandom(100, 255)})`;

			let xy = scale / 2 + i * scale + i * spread + x; //x coord of top left, (scale)/2 is the margin on the left added.
			let yy = scale / 2 + y + (1 - layerLeftSize * 1) * scale * 3; // y coord of top left, scale/2 is the top margin.
			let xyy = scale / 2 + i * scale + i * spread + x + scale; //x coord of top right, each layer is "scale" pixels wide
			let yyy = scale / 2 + y + (1 - layerRightSize * 1) * scale * 3; // y coord of top right corner, not sure why the sqrt here works LOL

			ctx.beginPath();
			ctx.moveTo(xy, yy);
			ctx.lineTo(xyy, yyy);
			ctx.lineTo(xyy, yyy + layerRightSize * scale * 6);
			ctx.lineTo(xy, yy + layerLeftSize * scale * 6);
			ctx.lineTo(xy, yy);
			ctx.fill();
			ctx.closePath();
			if (layer.constructor.name == "ResEmitterLayer" && layer.receiver) {
				ctx.fillStyle = `rgb(${Ment.seededRandomSP(0, 1, i * layerSize)},${Ment.seededRandomSP(
					0,
					255,
					i * layerRightSize * 69
				)},${Ment.seededRandomSP(0, 255, i * layerLeftSize * 420)})`;

				const heightVariance = Ment.seededRandomSP(2, 4, i * layerSize); // a random change in height so that hopefully arrows dont run into each other
				const height = scale / heightVariance + (1 - layerSize * 1) * scale * 3;
				const width = scale / 10;
				const st = scale / 2; //half of a scale
				ctx.fillRect(xy + st, yy - height, width, height);
				ctx.fillRect(xy + st, yy - height, scale + spread, width);
				for (var j = i; net.layers[j + 1].id != layer.id; j++) {
					ctx.fillRect(xy + (j - i + 1) * (scale + spread) + st, yy - height, scale + spread, width);
				}

				const receiverHeight =
					scale / heightVariance +
					(1 -
						(layer.receiver.inSize() > layer.receiver.outSize() ? layer.receiver.inSize() : layer.receiver.outSize()) / maxSize) *
						scale *
						3; // height of the line from the receiver box that was rendered to the top of the line
				//draw the arrow head
				const abovereceiver = yy - height + receiverHeight;
				const goofahh = (j - i + 1) * (scale + spread); //this made sense when I wrote it.
				ctx.fillRect(xy + goofahh + st, yy - height, width, receiverHeight - scale / 20);
				ctx.beginPath();
				ctx.moveTo(xy + goofahh + st + scale / 20, abovereceiver);
				ctx.lineTo(xy + goofahh + st / 1.5 + st + scale / 20, abovereceiver - scale / 4);
				ctx.lineTo(xy + goofahh - st / 1.5 + st + scale / 20, abovereceiver - scale / 4);
				ctx.moveTo(xy + goofahh + st + scale / 20, abovereceiver);
				ctx.fill();
				ctx.closePath();
			}
			layerSize = Math.max(0.5, layerSize);
			ctx.font = `${(layerSize * scale) / 2}px Arial`;
			ctx.fillStyle = "black";
			ctx.save();
			ctx.textAlign = "center";
			ctx.translate(xy + (xyy - xy) / 2, yyy + layerRightSize * scale * 3);
			ctx.rotate(-Math.PI / 2);
			ctx.fillText(layer.constructor.name, 0, 0);
			ctx.font = `${(layerSize * scale) / 3}px serif`;
			ctx.fillText(
				`(${layer.inSizeDimensions ? layer.inSizeDimensions() : layer.inSize()})(${
					layer.outSizeDimensions ? layer.outSizeDimensions() : layer.outSize()
				})`,
				0,
				scale / 3
			);
			ctx.restore();
			if (layer.outSizeDimensions && options.showAsImage) {
				wid = layer.outSizeDimensions()[0];
				hei = layer.outSizeDimensions()[1] | 1;
				dep = layer.outSizeDimensions()[2] | 1;
				for (var g = 0; g < dep; g++) {
					for (var h = 0; h < hei; h++) {
						for (var j = 0; j < wid; j++) {
							let row = scale * 6 + y + (scale / 25) * h;
							c = layer.outData[h * wid + j + g * (wid * hei)] * 255;
							ctx.fillStyle = "rgb(" + c + "," + c + "," + c + ")";
							ctx.fillRect(
								xy + -((hei * scale) / 25 - scale) / 2 + (scale / 25) * j,
								row + ((scale * hei) / 25) * g,
								scale / 25,
								scale / 25
							);
						}
					}
				}
			}
		}
	};

	let render = function (options) {
		let net = options.net;
		let ctx = options.ctx;
		let x = options.x;
		let y = options.y;
		let scale = options.scale;
		let spread = options.spread | 3; //pixel space between layers default is 3 pixels
		let background = options.background;

		let maxSize = 1;
		for (var i = 0; i < net.layers.length; i++) {
			if (net.layers[i].inSize() > maxSize) {
				maxSize = net.layers[i].inSize();
			} //end of if
			if (net.layers[i].outSize() > maxSize) {
				maxSize = net.layers[i].outSize();
			} //end of if
		}

		if (background == undefined || background == true) {
			background = "gray"; //default color
		}
		if (background) {
			ctx.fillStyle = background;
			ctx.fillRect(x, y, net.layers.length * (spread + scale) + scale, maxSize * scale * 2);
		}

		//render the neurons in each layer...
		for (i = 0; i < net.layers.length; i++) {
			let layer = net.layers[i];
			if (layer.constructor.name != "ConvLayer" && layer.constructor.name != "MaxPoolLayer") {
				for (var h = 0; h < layer.outSize(); h++) {
					ctx.fillStyle = "rgb(" + layer.outData[h] * 255 + "," + layer.outData[h] * 255 + "," + layer.outData[h] * 255 + ")";
					ctx.fillRect(
						x + (i + 1) * (scale + spread),
						y + h * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.outSize() * scale * 2) / 2),
						scale,
						scale
					);
				}
				if (i == 0) {
					for (var j = 0; j < layer.inSize(); j++) {
						ctx.fillStyle = "rgb(" + layer.inData[j] * 255 + "," + layer.inData[j] * 255 + "," + layer.inData[j] * 255 + ")";
						ctx.fillRect(
							x + i * (scale + spread),
							y + j * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2),
							scale,
							scale
						);
					}
				}
				//end of render neurons for each layer
			}
			if (layer.constructor.name == "FCLayer") {
				//render weights in the fc layer

				for (var j = 0; j < layer.inSize(); j++) {
					for (var h = 0; h < layer.outSize(); h++) {
						ctx.strokeStyle =
							"rgb(" + -(layer.w[j * layer.outSize() + h] * 255) + "," + layer.w[j * layer.outSize() + h] * 255 + ",0)";
						ctx.beginPath();
						ctx.moveTo(
							x + (i + 1) * (scale + spread) + scale / 2,
							y + h * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.outSize() * scale * 2) / 2) + scale / 2
						);
						ctx.lineTo(
							x + i * (scale + spread) + scale / 2,
							y + j * scale * 2 + ((maxSize * scale * 2) / 2 - (layer.inSize() * scale * 2) / 2) + scale / 2
						);
						ctx.stroke();
					} //end of foor loop each outSize
				} //end of for loop each inSize
			} //end of if layer is fc

			if (layer.constructor.name != "FCLayer") {
				//render the layer name
				ctx.fillStyle = "white";
				ctx.font = scale + "px serif";
				ctx.fillText(layer.constructor.name.slice(0, -5), x + i * (scale + spread) + scale * 1.5, y + (maxSize * scale * 2) / 2);
			}
		} //end of for loop each layer
	};

	function initWebMonkeys(monkeyObject) {
		Ment.webMonkeys = monkeyObject;
	}

	Ment.clamp = clamp;
	Ment.initWebMonkeys = initWebMonkeys;
	Ment.render = render;
	Ment.renderBox = renderBox;
	Ment.getLoss = getLoss;
	Ment.tanh = Math.tanh;
	Ment.lin = lin;
	Ment.bounce = bounce;
	Ment.isBrowser = isBrowser;
	Ment.makeid = makeid;
	Ment.gaussRandom = gaussRandom;
	Ment.polluteGlobal = polluteGlobal;
	Ment.protectNaN = protectNaN;
	Ment.inputError = inputError;
	Ment.printNet = printNet;

	let globalObject = isBrowser() ? window : global;

	if (!Ment.isBrowser()) {
		//test if we are in node
		module.exports = Ment; // in nodejs
	} else {
		window.Ment = Ment; // in ordinary browser attach library to window
		Ment.Worker = window.Worker;
	}
}

var Ment = Ment || {};
{
	load(this, function (exports) {
		function WebMonkeys(opt) {
			var maxVertexIndex,
				vertexIndexBuffer,
				resultTextureSide,
				resultTexture,
				arrays,
				arrayByName,
				shaderByTask,
				gl,
				defaultLib,
				writer,
				renderer,
				userLib,
				framebuffer,
				rendererVertexBuffer;

			// () -> Monkeys
			function init() {
				opt = opt || [];
				maxVertexIndex = 0;
				resultTextureSide = 0;
				arrays = [];
				arrayByName = {};
				shaderByTask = {};

				var glOpt = { antialias: false, preserveDrawingBuffer: true };

				var ENVIRONMENT_IS_WEB = typeof window === "object";
				var ENVIRONMENT_IS_WORKER = typeof importScripts === "function";
				var ENVIRONMENT_IS_NODE =
					typeof process === "object" && typeof require === "function" && !ENVIRONMENT_IS_WEB && !ENVIRONMENT_IS_WORKER;

				if (ENVIRONMENT_IS_NODE) {
					gl = require("g" + "l")(1, 1, glOpt);
				} else {
					var canvas;
					if (ENVIRONMENT_IS_WEB) {
						canvas = document.createElement("canvas");
					}
					if (ENVIRONMENT_IS_WORKER) {
						canvas = new OffscreenCanvas(1, 1);
					}
					gl = canvas.getContext("webgl", glOpt) || canvas.getContext("experimental-webgl", glOpt);
					gl.canvas = canvas;
					gl.canvas.width = 1;
					gl.canvas.height = 1;
					gl.canvas.style = [
						"border: 1px solid black;",
						"image-rendering: optimizeSpeed;",
						"image-rendering: -moz-crisp-edges;",
						"image-rendering: -webkit-optimize-contrast;",
						"image-rendering: -o-crisp-edges;",
						"image-rendering: pixelated;",
						"-ms-interpolation-mode: nearest-neighbor;",
					].join("");
				}

				defaultLib = [
					"vec2 indexToPos(vec2 size, float index){",
					"  return vec2(mod(index, size.x), floor(index/size.x));",
					"}",
					"float posToIndex(vec2 size, vec2 pos){",
					"  return pos.y*size.x + pos.x;",
					"}",
					"vec2 scaleRange(vec2 fromA, vec2 fromB, vec2 toA, vec2 toB, vec2 pos){",
					"  return toA+(pos-fromA)/(fromB-fromA)*(toB-toA);",
					"}",
					"vec4 packFloat(float x){",
					"  float s = 0.0;",
					"  float e = 0.0;",
					"  float m = x;",
					"  if (m<0.0) s=1.0, m=-m;",
					"  for (int i=0; i<24; ++i){",
					"    if (m>=2.0) m=m/2.0, e+=1.0;",
					"    if (m< 1.0) m=m*2.0, e-=1.0;",
					"    if (m>=1.0 && m<2.0) break;",
					"  };",
					"  return vec4(",
					"    floor(fract((m-1.0)*256.0*256.0)*256.0),",
					"    floor(fract((m-1.0)*256.0)*256.0),",
					"    floor(fract((m-1.0)*1.0)*256.0),",
					"    ((e+63.0) + (x>0.0?128.0:0.0)))/255.0;",
					"}",
					"float unpackFloat(vec4 v){",
					"  v *= 255.0;",
					"  float s = v.a >= 128.0 ? 1.0 : -1.0;",
					"  float e = v.a - (v.a >= 128.0 ? 128.0 : 0.0) - 63.0;",
					"  float m = 1.0 + v.x/256.0/256.0/256.0 + v.y/256.0/256.0 + v.z/256.0;",
					"  return s * pow(2.0, e) * m;",
					"}",
					"vec4 packVec4(vec4 v){",
					"  return v/255.0;",
					"}",
					"vec4 unpackVec4(vec4 v){",
					"  return v*255.0;",
					"}",
					"vec4 packIndexDepth(int a, int b){",
					"  float av = float(a);",
					"  float bv = float(b);",
					"  float x = mod(floor(av), 256.0);",
					"  float y = mod(floor(av/256.0), 256.0);",
					"  float z = mod(floor(av/256.0/256.0), 256.0);",
					"  float w = mod(floor(bv), 256.0);",
					"  return vec4(x,y,z,w)/255.0;",
					"}",
					"int unpackIndex(vec4 v){",
					"  return int(v.x*255.0 + v.y*255.0*256.0 + v.z*255.0*256.0*256.0);",
					"}",
					"int unpackDepth(vec4 v){",
					"  return int(v.w*255.0);",
					"}",
				].join("\n");

				writer = buildShader(
					[
						"precision highp float;",
						"attribute float resultIndex;",
						"uniform sampler2D resultTexture;",
						"uniform float resultTextureSide;",
						"uniform float resultGridSide;",
						"uniform float resultSquareSide;",
						"uniform float targetTextureSide;",
						"varying vec4 value;",
						defaultLib,
						"void main(){",
						"  float resultSquareIndex = mod(resultIndex, resultSquareSide*resultSquareSide/2.0);",
						"  vec2 resultSquareCoord = indexToPos(vec2(resultSquareSide/2.0,resultSquareSide), resultSquareIndex)*vec2(2.0,1.0);",
						"  vec2 resultGridCoord = indexToPos(vec2(resultGridSide), floor(resultIndex/(resultSquareSide*resultSquareSide/2.0)));",
						"  vec2 resultCoord = resultGridCoord * resultSquareSide + resultSquareCoord;",
						"  vec2 indexCoord = (resultCoord+vec2(0.5,0.5))/resultTextureSide;",
						"  vec2 valueCoord = (resultCoord+vec2(1.5,0.5))/resultTextureSide;",
						"  float index = float(unpackIndex(texture2D(resultTexture, indexCoord))-1);",
						"  float depth = float(unpackDepth(texture2D(resultTexture, indexCoord)));",
						"  value = texture2D(resultTexture, valueCoord);",
						"  vec2 rPos = (indexToPos(vec2(targetTextureSide),index)+vec2(0.5))/targetTextureSide*2.0-1.0;",
						"  gl_Position = vec4(depth > 0.5 ? rPos : vec2(-1.0,-1.0), (255.0-depth)/255.0, 1.0);",
						//"  gl_Position = vec4(rPos, -0.5, 1.0);",
						"  gl_PointSize = 1.0;",
						"}",
					].join("\n"),
					["precision highp float;", "varying vec4 value;", "void main(){", "  gl_FragColor = value;", "}"].join("\n")
				);

				renderer = buildShader(
					[
						"precision highp float;",
						"attribute vec2 vertexPos;",
						"varying vec2 pos;",
						"void main(){",
						"  pos = vertexPos;",
						"  gl_Position = vec4(vertexPos, 0.0, 1.0);",
						"}",
					].join("\n"),
					[
						"precision mediump float;",
						"uniform sampler2D array;",
						"varying vec2 pos;",
						"void main(){",
						"  gl_FragColor = texture2D(array, pos*0.5+0.5);",
						"}",
					].join("\n")
				);

				gl.clearDepth(256.0);

				vertexIndexBuffer = gl.createBuffer();

				rendererVertexBuffer = gl.createBuffer();
				gl.bindBuffer(gl.ARRAY_BUFFER, rendererVertexBuffer);
				gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1]), gl.STATIC_DRAW);

				resultTexture = gl.createTexture();
				gl.activeTexture(gl.TEXTURE0);
				gl.bindTexture(gl.TEXTURE_2D, resultTexture);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

				framebuffer = gl.createFramebuffer();
				gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

				return monkeysApi;
			}

			// *Monkeys => Number -> Monkeys
			//   Makes sure there are enough index vertices available
			//   for a `gl.drawArrays(gl.POINTS, 0, vertices)` call.
			function allocVertexIndices(indices) {
				if (indices > maxVertexIndex) {
					maxVertexIndex = Math.pow(fitTextureSide(indices), 2);
					var vertexIndexArray = new Float32Array(maxVertexIndex);
					for (var i = 0; i < maxVertexIndex; ++i) vertexIndexArray[i] = i;
					gl.bindBuffer(gl.ARRAY_BUFFER, vertexIndexBuffer);
					gl.bufferData(gl.ARRAY_BUFFER, vertexIndexArray, gl.STATIC_DRAW);
					gl.bindBuffer(gl.ARRAY_BUFFER, null);
				}
			}

			// *Monkeys => Number -> ()
			//   Makes sure the results texture is big
			//   enough to fit every result of a task.
			function allocResultTexture(usedTextureSide) {
				if (usedTextureSide > resultTextureSide) {
					resultTextureSide = usedTextureSide;
					gl.activeTexture(gl.TEXTURE0);
					gl.bindTexture(gl.TEXTURE_2D, resultTexture);
					gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, resultTextureSide, resultTextureSide, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
				}
			}

			// *Monkeys => String, String -> WebGLProgram
			function buildShader(vertexSrc, fragmentSrc) {
				function compile(type, shaderSource) {
					var shader = gl.createShader(type);
					gl.shaderSource(shader, shaderSource);
					gl.compileShader(shader);
					if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
						var errorMsg = "WebMonkeys had the following error from WebGL: " + gl.getShaderInfoLog(shader);
						if (errorMsg.indexOf("syntax error") !== -1) errorMsg += "This could be fixed by adding extra `;` before setters.";
						throw errorMsg;
					}
					return shader;
				}
				var vertexShader = compile(gl.VERTEX_SHADER, vertexSrc);
				var fragmentShader = compile(gl.FRAGMENT_SHADER, fragmentSrc);

				var shader = gl.createProgram();
				gl.attachShader(shader, vertexShader);
				gl.attachShader(shader, fragmentShader);
				gl.linkProgram(shader);
				if (!gl.getProgramParameter(shader, gl.LINK_STATUS)) throw "Error linking shaders.";

				return shader;
			}

			// Number -> Number
			function fitTextureSide(elements) {
				return Math.pow(2, Math.ceil(Math.log(Math.sqrt(elements)) / Math.log(2)));
			}

			// Number -> Number
			function fract(x) {
				return x - Math.floor(x);
			}

			// *Monkeys => String -> Maybe (Either (Array Number) *Uint32Array)
			function get(name) {
				var array = arrayByName[name];
				if (!array) return null;
				var targetArray = array.uint32Array;
				var pixels = targetArray
					? new Uint8Array(targetArray.buffer) // re-uses existing buffer
					: new Uint8Array(array.textureSide * array.textureSide * 4);
				gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, array.texture, 0);
				gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, null);
				gl.readPixels(0, 0, array.textureSide, array.textureSide, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

				if (!targetArray) {
					var result = new Float32Array(array.length);
					for (var i = 0, l = array.length; i < l; ++i) {
						var s = pixels[i * 4 + 3] >= 128 ? 1 : -1;
						var e = pixels[i * 4 + 3] - (pixels[i * 4 + 3] >= 128 ? 128 : 0) - 63;
						var m = 1 + pixels[i * 4 + 0] / 256 / 256 / 256 + pixels[i * 4 + 1] / 256 / 256 + pixels[i * 4 + 2] / 256;
						var n = s * Math.pow(2, e) * m;
						var z = 0.000000000000000001; // to avoid annoying floating point error for 0
						// result.push(-z < n && n < z ? 0 : n); //floating point error is miniscule in ai computation.
						result[i] = n;
					}
					return result;
				} else {
					return targetArray;
				}
			}

			function getLen(name) {
				var array = arrayByName[name];
				return array.length;
			}

			// *Monkeys => String, *Uint32Array -> Monkeys
			// *Monkeys => String, Array Number -> Monkeys
			// *Monkeys => String, Number -> Monkeys
			function set(name, lengthOrArray) {
				if (typeof lengthOrArray === "number") {
					var length = lengthOrArray;
					var textureSide = fitTextureSide(length);
					var array = null;
				} else {
					var length = lengthOrArray.length;
					var textureSide = fitTextureSide(length);
					if (lengthOrArray instanceof Array || lengthOrArray instanceof Float32Array || lengthOrArray instanceof Float64Array) {
						// upload JS Numbers as Floats
						var array = new Uint8Array(textureSide * textureSide * 4);
						for (var i = 0, l = lengthOrArray.length; i < l; ++i) {
							var x = lengthOrArray[i];
							var s = x > 0 ? 1 : -1;
							var e = Math.floor(Math.log(s * x) / Math.LN2);
							var m = (s * x) / Math.pow(2, e);
							array[i * 4 + 0] = Math.floor(fract((m - 1) * 256 * 256) * 256) || 0;
							array[i * 4 + 1] = Math.floor(fract((m - 1) * 256) * 256) || 0;
							array[i * 4 + 2] = Math.floor(fract((m - 1) * 1) * 256) || 0;
							array[i * 4 + 3] = e + 63 + (x > 0 ? 128 : 0) || 0;
						}
					} else {
						// upload 32-bit Uints as Vec4s
						if (textureSide * textureSide !== length)
							throw (
								"WebMonkey error: when on raw buffer mode, the length of your\n" +
								"buffer must be (2^n)^2 for a positive integer n. That is, it\n" +
								"could be 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144\n" +
								"and so on. Your '" +
								name +
								"' buffer has length " +
								length +
								"."
							);
						var array = new Uint8Array(lengthOrArray.buffer);
					}
				}
				gl.activeTexture(gl.TEXTURE0);
				if (!arrayByName[name]) {
					var texture = gl.createTexture();
					gl.bindTexture(gl.TEXTURE_2D, texture);
					gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
					gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
					gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
					gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
					gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, textureSide, textureSide, 0, gl.RGBA, gl.UNSIGNED_BYTE, array);
					var depthbuffer = gl.createRenderbuffer();
					gl.bindRenderbuffer(gl.RENDERBUFFER, depthbuffer);
					gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, textureSide, textureSide);
					arrayByName[name] = {
						name: name,
						uint32Array: lengthOrArray instanceof Uint32Array ? lengthOrArray : null,
						valueType: lengthOrArray instanceof Uint32Array ? "vec4" : "float",
						texture: texture,
						depthbuffer: depthbuffer,
						textureName: name + "_",
						textureSide: textureSide,
						length: length,
					};
					arrays.push(arrayByName[name]);
				} else {
					var texture = arrayByName[name].texture;
					gl.bindTexture(gl.TEXTURE_2D, texture);
					gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, textureSide, textureSide, 0, gl.RGBA, gl.UNSIGNED_BYTE, array);
				}
				return monkeysApi;
			}

			// *Monkeys => String, Number -> Monkeys
			//   Fills an array with a floating point number
			function fill(name, x) {
				var array = arrayByName[name];
				// Since the float packing on the set function is
				// inlined for performance, it must be duplicated
				// here. FIXME: find a way to avoid this.
				var s = x > 0 ? 1 : -1;
				var e = Math.floor(Math.log2(s * x));
				var m = (s * x) / Math.pow(2, e);
				var a = Math.floor(fract((m - 1) * 256 * 256) * 256) || 0;
				var b = Math.floor(fract((m - 1) * 256) * 256) || 0;
				var c = Math.floor(fract((m - 1) * 1) * 256) || 0;
				var d = e + 63 + (x > 0 ? 128 : 0) || 0;
				return clear(name, (d << 24) + (c << 16) + (b << 8) + a);
			}

			// *Monkeys => String, Uint32 -> Monkeys
			//   Fills an array with an Uint32
			function clear(name, value) {
				var array = arrayByName[name];
				gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
				gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, array.texture, 0);
				gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, null);
				gl.clearColor(
					((value & 0x000000ff) >>> 0) / 255,
					((value & 0x0000ff00) >>> 8) / 255,
					((value & 0x00ff0000) >>> 16) / 255,
					((value & 0xff000000) >>> 24) / 255
				);
				gl.clear(gl.COLOR_BUFFER_BIT);
				return monkeysApi;
			}

			// *Monkeys => String -> Monkeys
			function del(name) {
				var existingArray;
				if ((existingArray = arrayByName[name])) {
					delete arrayByName[name];
					arrays = arrays.filter(function (arr) {
						return arr !== existingArray;
					});
					gl.deleteTexture(existingArray.texture);
				}
				return monkeysApi;
			}

			// String -> Maybe {name: String, index: String, depth: String, value: String}
			//   Parses a setter statement such as `foo(i*8) := bar(i*8) + baz(i*8);` and
			//   returns `name`, `index`, `depth` and `value` strings:
			//   {name: "foo", index: "i*8", depth: "", value: "bar(i*8) + baz(i*8)"}
			function parseSetterStatement(statement) {
				var name = "";
				var index = "";
				var depth = "";
				var value = "";
				var phase = 0;
				var brackets = 1;
				for (var i = 0, l = statement.length; i < l; ++i) {
					var chr = statement[i];
					switch (phase) {
						case 0:
							if (chr === "(") phase = 1;
							else if (chr !== " " && chr !== "\n") name += chr;
							break;
						case 1:
							if (chr === "(") ++brackets;
							else if (chr === ")") --brackets;
							if (brackets === 1 && chr === ",") phase = 2;
							else if (brackets === 0) phase = 3;
							else index += chr;
							break;
						case 2:
							if (chr === "(") ++brackets;
							else if (chr === ")") --brackets;
							if (brackets === 0) phase = 3;
							else depth += chr;
							break;
						case 3:
							if (chr === ":") phase = 4;
							break;
						case 4:
							if (chr === "=") phase = 5;
							else return null;
							break;
						case 5:
							if (chr !== " ") (value += chr), (phase = 6);
							break;
						case 6:
							if (chr === ";") phase = 7;
							else value += chr;
							break;
					}
				}
				return phase === 7 ? { name: name, index: index, depth: depth, value: value } : null;
			}

			// String -> {
			//   shader: GLShader,
			//   usedResults: Number,
			//   allocResults: Number,
			//   resultArrayName: String,
			//   usesDepth: Bool}
			function buildTask(task) {
				if (shaderByTask[task]) return shaderByTask[task];

				var usesDepth = false;
				var taskStatements = task.split(";");
				taskStatements.pop();
				var setters = [];
				var setter;
				while ((setter = parseSetterStatement(taskStatements[taskStatements.length - 1] + ";"))) {
					setters.push(setter);
					taskStatements.pop();
					if (setter.depth !== "0") usesDepth = true;
				}
				if (setters.length === 0)
					throw "Error parsing Monkey task: tasks must end with a setter statement such as `foo[0] = 0;`.";
				var resultArrayName = setters[0].name;
				for (var i = 1, l = setters.length; i < l; ++i)
					if (setters[i].name !== resultArrayName)
						throw "Error parsing Monkey task: you can't write to different arrays on the same task.";

				var taskWithoutSetters = taskStatements.join(";") + ";";

				// `usedResults` is how many sets this work does.
				// `allocResults` is how many sets we actually allocated space for.
				// Explanation: a result is an (indice, value) pair which will be used on
				// the next pass to fill the target array. Those results are recorded
				// into square sections of a 2D texture. Each monkey has its own square.
				// In order for everything to fit, the square of a monkey will have empty
				// space. For example, if a task makes 3 sets, it requires 6 pixels on
				// the texture report its result (3 indices + 3 values). To fit 6 pixels,
				// we need a square of side 4; side 2 isn't enough because it only fits 4
				// pixels, side 3 isn't allowed because such a square wouldn't align
				// correctly on the texture.
				// TODO: complete this explanation, move it to the top, make some drawings
				var usedResults = setters.length;
				var allocResults = Math.pow(fitTextureSide(usedResults * 2), 2) / 2;

				var getters = "";
				for (var i = 0, l = arrays.length; i < l; ++i)
					getters +=
						"uniform sampler2D " +
						arrays[i].textureName +
						";\n" +
						arrays[i].valueType +
						" " +
						arrays[i].name +
						"(float idx){\n" +
						"  return " +
						(arrays[i].valueType === "float" ? "unpackFloat" : "unpackVec4") +
						"(texture2D(" +
						arrays[i].textureName +
						",indexToPos(vec2(" +
						arrays[i].textureSide.toFixed(1) +
						"), idx)/" +
						arrays[i].textureSide.toFixed(2) +
						"));\n" +
						"}\n" +
						arrays[i].valueType +
						" " +
						arrays[i].name +
						"(int idx){\n" +
						"  return " +
						arrays[i].name +
						"(float(idx));\n" +
						"}\n";

				var setterFns = "";
				for (var i = 0; i < allocResults; ++i) {
					setterFns += "void set" + i + "(int i" + i + ", int d" + i + ", float v" + i + "){\n";
					setterFns += "  results[" + (i * 2 + 0) + "] = packIndexDepth(i" + i + "+1, d" + i + ");\n";
					setterFns += "  results[" + (i * 2 + 1) + "] = packFloat(v" + i + ");\n";
					setterFns += "}\n";
					setterFns += "void set" + i + "(int i" + i + ", int d" + i + ", vec4 v" + i + "){\n";
					setterFns += "  results[" + (i * 2 + 0) + "] = packIndexDepth(i" + i + "+1, d" + i + ");\n";
					setterFns += "  results[" + (i * 2 + 1) + "] = packVec4(v" + i + ");\n";
					setterFns += "}\n";
				}

				var writeToTexture = "";
				for (var i = 0; i < allocResults * 2; ++i)
					writeToTexture += "  if (idx == " + i + ") gl_FragColor = results[" + i + "];\n";

				var setter = "";
				for (var i = 0; i < allocResults; ++i) {
					setter += "  set" + i + "(";
					setter +=
						i < usedResults ? setters[i].index + ", " + (setters[i].depth || "1") + ", " + setters[i].value : "0, 0, vec4(0.0)";
					setter += ");\n";
				}

				var vertexShader = [
					"precision highp float;",
					"uniform float resultTextureSide;",
					"uniform float resultGridSide;",
					"uniform float resultSquareSide;",
					"attribute float resultIndex;",
					"varying float resultIndexVar;",
					"varying vec4 results[" + allocResults * 2 + "];",
					defaultLib,
					getters,
					setterFns,
					userLib,
					"vec4 scaleToScreen(vec2 pos){",
					"  vec2 screenCoord = scaleRange(vec2(0.0,0.0), vec2(resultGridSide), vec2(-1.0), vec2(-1.0+resultSquareSide*resultGridSide/resultTextureSide*2.0), pos);",
					"  return vec4(screenCoord + vec2(resultSquareSide)/resultTextureSide, 1.0, 1.0);",
					"}",
					"void main(){",
					"  int i = int(resultIndex);",
					"  float f = resultIndex;",
					taskWithoutSetters,
					setter,
					"  gl_PointSize = resultSquareSide;",
					"  gl_Position = scaleToScreen(indexToPos(vec2(resultGridSide), resultIndex));",
					"  resultIndexVar = resultIndex;",
					"}",
				].join("\n");

				var fragmentShader = [
					"precision highp float;",
					"varying float resultIndexVar;",
					"varying vec4 results[" + allocResults * 2 + "];",
					"uniform float resultSquareSide;",
					defaultLib,
					"void main(){",
					"  vec2 coord = floor(gl_PointCoord * resultSquareSide);",
					"  int idx = int((resultSquareSide-1.0-coord.y) * resultSquareSide + coord.x);",
					writeToTexture,
					"}",
				].join("\n");

				var shader = buildShader(vertexShader, fragmentShader);

				return (shaderByTask[task] = {
					usesDepth: usesDepth,
					usedResults: usedResults,
					allocResults: allocResults,
					shader: shader,
					resultArrayName: resultArrayName,
				});
			}

			// *Monkeys => Number, String -> Monkeys
			function work(monkeyCount, taskSource) {
				var task = buildTask(taskSource);
				var resultArray = arrayByName[task.resultArrayName];
				var resultSquareSide = fitTextureSide(task.allocResults * 2);
				var resultGridSide = fitTextureSide(monkeyCount);
				var usedResultTextureSide = resultGridSide * resultSquareSide;
				var resultSquareSide = fitTextureSide(task.allocResults * 2);
				var resultGridSide = fitTextureSide(monkeyCount);
				var usedResultTextureSide = resultGridSide * resultSquareSide;

				allocResultTexture(usedResultTextureSide);
				allocVertexIndices(Math.max(monkeyCount, (monkeyCount * resultSquareSide * resultSquareSide) / 2));

				gl.useProgram(task.shader);
				gl.bindBuffer(gl.ARRAY_BUFFER, vertexIndexBuffer);
				gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
				gl.uniform1f(gl.getUniformLocation(task.shader, "resultGridSide"), resultGridSide);
				gl.uniform1f(gl.getUniformLocation(task.shader, "resultSquareSide"), resultSquareSide);
				gl.uniform1f(gl.getUniformLocation(task.shader, "resultTextureSide"), resultTextureSide);
				gl.vertexAttribPointer(gl.getAttribLocation(task.shader, "resultIndex"), 1, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(gl.getAttribLocation(task.shader, "resultIndex"));
				gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, resultTexture, 0);
				gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, null);
				gl.viewport(0, 0, resultTextureSide, resultTextureSide);
				for (var i = 0, l = arrays.length; i < l; ++i) {
					gl.activeTexture(gl.TEXTURE0 + i);
					gl.bindTexture(gl.TEXTURE_2D, arrays[i].texture);
					gl.uniform1i(gl.getUniformLocation(task.shader, arrays[i].textureName), i);
				}
				gl.drawArrays(gl.POINTS, 0, monkeyCount);

				if (task.usesDepth) gl.enable(gl.DEPTH_TEST);
				gl.useProgram(writer);
				gl.activeTexture(gl.TEXTURE0);
				gl.bindTexture(gl.TEXTURE_2D, resultTexture);
				gl.uniform1i(gl.getUniformLocation(writer, "resultTexture"), resultTexture);
				gl.uniform1f(gl.getUniformLocation(writer, "resultGridSide"), resultGridSide);
				gl.uniform1f(gl.getUniformLocation(writer, "resultSquareSide"), resultSquareSide);
				gl.uniform1f(gl.getUniformLocation(writer, "resultTextureSide"), resultTextureSide);
				gl.uniform1f(gl.getUniformLocation(writer, "targetTextureSide"), resultArray.textureSide);
				gl.vertexAttribPointer(gl.getAttribLocation(writer, "resultIndex"), 1, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(gl.getAttribLocation(writer, "resultIndex"));
				gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, resultArray.texture, 0);
				gl.viewport(0, 0, resultArray.textureSide, resultArray.textureSide);
				if (task.usesDepth) {
					gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, resultArray.depthbuffer);
					gl.clear(gl.DEPTH_BUFFER_BIT);
				}
				gl.drawArrays(gl.POINTS, 0, (monkeyCount * resultSquareSide * resultSquareSide) / 2);
				if (task.usesDepth) gl.disable(gl.DEPTH_TEST);
				return monkeysApi;
			}

			// Allows rendering arrays to a Canvas for visualization
			// *Monkeys => String, Number, Number -> Maybe Canvas
			function render(name, width, height) {
				if (gl.canvas && arrayByName[name]) {
					gl.canvas.width = width;
					gl.canvas.height = height;
					gl.useProgram(renderer);
					gl.viewport(0, 0, width, height);

					gl.activeTexture(gl.TEXTURE0);
					gl.bindTexture(gl.TEXTURE_2D, arrayByName[name].texture);

					gl.bindBuffer(gl.ARRAY_BUFFER, rendererVertexBuffer);
					var vertexPosAttr = gl.getAttribLocation(renderer, "vertexPos");
					gl.vertexAttribPointer(vertexPosAttr, 2, gl.FLOAT, false, 0, 0);
					gl.enableVertexAttribArray(vertexPosAttr);
					gl.bindFramebuffer(gl.FRAMEBUFFER, null);

					gl.drawArrays(gl.TRIANGLES, 0, 6);
					return gl.canvas;
				}
				return null;
			}

			// *Monkeys => String -> Monkeys
			function lib(source) {
				userLib = source;
				return monkeysApi;
			}

			// Monkeys => String -> String
			function stringify(name) {
				return JSON.stringify(get(name));
			}

			// Monkeys => String -> IO ()
			function log(name) {
				console.log(stringify(name));
			}

			var monkeysApi = {
				set: set,
				get: get,
				getLen: getLen,
				del: del,
				lib: lib,
				work: work,
				clear: clear,
				fill: fill,
				render: render,
				stringify: stringify,
				log: log,
			};

			return init();
		}

		if (typeof window === "object") exports.WebMonkeys = WebMonkeys;

		if (typeof module !== "undefined") module.exports = WebMonkeys;
	});

	function load(root, factory) {
		"use strict";

		// amd
		if (typeof define === "function" && define.amd)
			// register as an anonymous module
			define([], factory);
		// commonjs
		else if (typeof exports === "object" && typeof exports.nodeName !== "string") factory(exports);
		// browser globals
		else factory(root);
	}
	Ment.webMonkeys = WebMonkeys();
}

//Base layer for all activations to inherit from

{
	class ActivationBase {
		constructor(size) {
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(size);
			this.outData = new Float32Array(size);
			this.costs = new Float32Array(size); //costs for each neuron
			this.pl; //reference to previous layer
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

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		forward(inData, actFunction) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw Ment.inputError(this, inData);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = actFunction(this.inData[h]);
			}
			//Oh the misery
		}

		backward(err, actFunctionPrime) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			for (var j = 0; j < this.outSize(); j++) {
				this.costs[j] = err[j] * actFunctionPrime(this.inData[j]);
			}
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;

			return ret;
		}
	}

	Ment.ActivationBase = ActivationBase;
}

{
	class AveragePoolLayer {
		constructor(inDim, filterDim, stride = 1) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			if (filterDim.length != 2) {
				throw (
					this.constructor.name +
					" parameter error: Missing filter dimensions parameter. \n" +
					"First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)"
				);
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];

			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.outData = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.inDepth
			);
			this.inData = new Float32Array(inWidth * inHeight * inDepth);
			this.costs = new Float32Array(inWidth * inHeight * inDepth);
			this.maxIndexes = new Float32Array(this.outData.length);
			this.accessed = new Float32Array(this.costs.length).fill(1);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Average Pool layer error: Pooling size (width / height) cannot be bigger than the inputs corresponding (width/height)";
			}

			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//Change these for broken code!
			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.inDepth,
			];
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

			for (var g = 0; g < this.hMFHPO; g++) {
				const ga = g * this.stride;
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					const ba = b * this.stride;
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						const hWIH = h * this.wIH + ba;
						let avg = 0;
						for (var j = 0; j < this.filterHeight; j++) {
							const jGAIWBA = (j + ga) * this.inWidth + hWIH;
							for (var k = 0; k < this.filterWidth; k++) {
								avg += this.inData[k + jGAIWBA];
							}
						}
						this.outData[odi] = avg / (this.filterWidth * this.filterHeight);
					}
				}
			}
		}

		backward(err) {
			this.costs.fill(0);
			if (!err) {
				err = this.nextLayer.costs;
			}
			for (var g = 0; g < this.hMFHPO; g++) {
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						this.costs[this.maxIndexes[odi]] += err[odi];
						this.accessed[this.maxIndexes[odi]]++;
					}
				}
			}
			for (var g = 0; g < this.hMFHPO; g++) {
				const ga = g * this.stride;
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					const ba = b * this.stride;
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						const hWIH = h * this.wIH + ba;
						for (var j = 0; j < this.filterHeight; j++) {
							const jGAIWBA = (j + ga) * this.inWidth + hWIH;
							for (var k = 0; k < this.filterWidth; k++) {
								this.costs[k + jGAIWBA] += err[odi];
								this.accessed[k + jGAIWBA] += 1;
							}
						}
					}
				}
			}
			for (var i = 0; i < this.costs.length; i++) {
				this.costs[i] /= this.accessed[i] + this.filterWidth * this.filterHeight; //Average the grad
				this.accessed[i] = 0;
			}
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, stride = 1,
			let saveObject = JSON.parse(json);
			let layer = new AveragePoolLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.stride
			);
			return layer;
		}
	}

	Ment.AveragePoolLayer = AveragePoolLayer;
	Ment.AveragePool = AveragePoolLayer;
	Ment.AveragePooling = AveragePoolLayer;
	Ment.AvgPool = AveragePoolLayer;
	Ment.AvgPoolLayer = AveragePoolLayer;
	Ment.AvgPoolingLayer = AveragePoolLayer;
}

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
			for (var i = 0; i < this.b.length; i++) {
				this.b[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
			}
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
				this.b = new Float32Array(layer.outSize());
				this.bs = new Float32Array(layer.outSize());
				for (var i = 0; i < this.b.length; i++) {
					this.b[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
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

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			for (var j = 0; j < this.outData.length; j++) {
				this.costs[j] = err[j];
				this.bs[j] += err[j];
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

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "pl" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
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

		getParamsAndGrads(forUpdate = true) {
			return [this.b, this.bs];
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

{
	/*
if filter is 3x3 with indepth of 3 the first filter stored as such:
this.filterw = [0,1,2,3,4,5,6,7,8,
								0,1,2,3,4,5,6,7,8,
								0,1,2,3,4,5,6,7,8.......]

Each filter has depth equal to the InDepth.

this.filterw
[filterNum * this.inDepth * filterWidth * filterHeight + x + (y * filterWidth) + (depth * filterWidth * filterHeight)]

if you wanna input an image do it like this.
[r,r,r,r,g,g,g,g,b,b,b,b,b]
NOT like this:
[r,g,b,r,g,b,r,g,b,r,g,b]
Im sorry but I had to choose one 
*/

	class ConvLayer {
		static averageOutCosts = false; //probs
		static averageOutGrads = false; //ehhh
		constructor(inDim, filterDim, filters = 3, stride = 1, bias = true) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			if (filterDim.length != 2) {
				throw (
					this.constructor.name +
					" parameter error: Missing filter dimensions parameter. \n" +
					"First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)"
				);
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];
			this.filters = filters; //the amount of filters
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.filterw = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			this.filterws = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			this.outData = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			this.inData = new Float32Array(inWidth * inHeight * inDepth);
			this.accessed = new Float32Array(this.inData.length).fill(0); //to average out the costs

			this.inData.fill(0); //to prevent mishap
			this.costs = new Float32Array(inWidth * inHeight * inDepth);
			this.b = new Float32Array(this.outData.length);
			this.bs = new Float32Array(this.outData.length);
			this.useBias = bias;
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Conv layer error: filters cannot be bigger than the input";
			}
			//init random weights
			for (var i = 0; i < this.filterw.length; i++) {
				this.filterw[i] = 0.5 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
			}
			for (var a = 0; a < 3; a++) {
				let newFilterw = this.filterw.slice(0);
				for (var f = 0; f < this.filters; f++) {
					for (var d = 0; d < this.inDepth; d++) {
						for (var x = 0; x < this.filterWidth; x++) {
							for (var y = 0; y < this.filterHeight; y++) {
								let count = 0;
								let ind = [f * this.inDepth * filterWidth * filterHeight + x + y * filterWidth + d * filterWidth * filterHeight];
								let indR = [
									f * this.inDepth * filterWidth * filterHeight + (x + 1) + y * filterWidth + d * filterWidth * filterHeight,
								];
								let indL = [
									f * this.inDepth * filterWidth * filterHeight + (x - 1) + y * filterWidth + d * filterWidth * filterHeight,
								];
								let indD = [
									f * this.inDepth * filterWidth * filterHeight + x + (y + 1) * filterWidth + d * filterWidth * filterHeight,
								];
								let indU = [
									f * this.inDepth * filterWidth * filterHeight + x + (y - 1) * filterWidth + d * filterWidth * filterHeight,
								];
								if (x < filterWidth - 1) count += this.filterw[indR];
								if (x > 1) count += this.filterw[indL];
								if (y < filterHeight - 1) count += this.filterw[indD];
								if (y > 1) count += this.filterw[indU];
								newFilterw[ind] += count / 5;
							}
						}
					}
				}
				this.filterw = newFilterw;
			}
			if (this.useBias) {
				for (var i = 0; i < this.b.length; i++) {
					this.b[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
			} else {
				this.b.fill(0);
			}
			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//DONT CHANGE THESE OR BIG BREAKY BREAKY!

			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
			this.wIHID = this.inWidth * this.inHeight * this.inDepth;
			this.fWIH = this.filterWidth * this.filterHeight;
			this.fWIHID = this.inDepth * this.filterHeight * this.filterWidth;
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.filters,
			];
		}

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw inputError(this, inData);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}
			this.outData.fill(0);

			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * this.stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * this.stride;
						for (var h = 0; h < this.inDepth; h++) {
							const hWIH = h * this.wIH + ba;
							const hFWIH = h * this.fWIH + iFWIHID;
							for (var j = 0; j < this.filterHeight; j++) {
								const jGAIWBA = (j + ga) * this.inWidth + hWIH;
								const jFWHFWIH = j * this.filterWidth + hFWIH;
								for (var k = 0; k < this.filterWidth; k++) {
									this.outData[odi] += this.inData[k + jGAIWBA] * this.filterw[k + jFWHFWIH];
								}
							}
						}
						this.outData[odi] += this.b[odi];
					}
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			this.costs.fill(0); //reset the costs
			//-----------------------------Beginning of monstrosity-----------------
			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * this.stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * this.stride;
						for (var h = 0; h < this.inDepth; h++) {
							const hWIH = h * this.wIH;
							const hFWIH = h * this.fWIH + iFWIHID;
							for (var j = 0; j < this.filterHeight; j++) {
								const jGAIWBA = (j + ga) * this.inWidth + hWIH + ba;
								const jFWHFWIH = j * this.filterWidth + hFWIH;
								for (var k = 0; k < this.filterWidth; k++) {
									this.costs[k + jGAIWBA] += this.filterw[k + jFWHFWIH] * err[odi];
									this.accessed[k + jGAIWBA]++;
									this.filterws[k + jFWHFWIH] += this.inData[k + jGAIWBA] * err[odi];
								}
							}
						}
					}
				}
			}
			//---------------------------------End of monstrosity-----------------
			for (var i = 0; i < this.outData.length; i++) {
				this.bs[i] += err[i];
			}
			if (ConvLayer.averageOutCosts) {
				for (var i = 0; i < this.costs.length; i++) {
					this.costs[i] /= this.accessed[i];
					this.accessed[i] = 0;
				}
			}
		}

		getParamsAndGrads(forUpdate = true) {
			if (forUpdate) {
				if (ConvLayer.averageOutGrads) {
					for (var i = 0; i < this.filterws.length; i++) {
						this.filterws[i] /= this.filters;
					}
				}
			}
			if (this.useBias) {
				return [this.filterw, this.filterws, this.b, this.bs];
			} else {
				return [this.filterw, this.filterws];
			}
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "filterws" ||
					key == "filterbs" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "accessed" ||
					key == "bs" ||
					key == "ws" ||
					key == "hMFHPO" ||
					key == "wMFWPO" ||
					key == "hMFWMF" ||
					key == "wIH" ||
					key == "wIHID" ||
					key == "fWIH" ||
					key == "fWIHID" ||
					key == (this.useBias ? null : "b")
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ConvLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.filters,
				saveObject.stride,
				saveObject.useBias
			);
			for (var i = 0; i < layer.filterw.length; i++) {
				layer.filterw[i] = saveObject.filterw[i];
			}
			if (saveObject.useBias) {
				for (var i = 0; i < layer.b.length; i++) {
					layer.b[i] = saveObject.b[i];
				}
			}
			return layer;
		}
	}

	Ment.ConvLayer = ConvLayer;
	Ment.Conv = ConvLayer;
}

{
	class DataBlockLayer {
		//this layer outputs the inputs with no changes
		constructor(size) {
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(size); //the inData
			this.outData = new Float32Array(size); //will be init when "connect" is called.
			this.costs = new Float32Array(size); //costs for each neuron
			this.pl; //reference to previous layer
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

			this.outData.fill(0);
		}

		backward(err) {
			this.costs.fill(0); //best layer
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
					key == "pl" ||
					key == "outData" ||
					key == "costs" ||
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

{
	class DeconvLayer {
		static averageOutCosts = false;
		static averageOutGrads = false;
		constructor(inDim, filterDim, filters = 3, stride = 1, useBias = true) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			if (filterDim.length != 2) {
				throw (
					this.constructor.name +
					" parameter error: Missing filter dimensions parameter. \n" +
					"First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)"
				);
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];
			this.useBias = useBias;

			this.filters = filters; //the amount of filters
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.filterw = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			this.filterws = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			this.inData = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			this.outData = new Float32Array(inWidth * inHeight * inDepth);
			this.inData.fill(0); //to prevent mishap
			this.outData.fill(0); //to prevent mishap
			this.costs = new Float32Array(this.inData.length);
			this.b = new Float32Array(this.outData.length);
			this.bs = new Float32Array(this.outData.length);
			this.accessed = new Float32Array(this.inData.length).fill(0);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Conv layer error: filters cannot be bigger than the input";
			}
			//init random weights
			for (var i = 0; i < this.filterw.length; i++) {
				this.filterw[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
			}
			for (var a = 0; a < 3; a++) {
				let newFilterw = this.filterw.slice(0);
				for (var f = 0; f < this.filters; f++) {
					for (var d = 0; d < this.inDepth; d++) {
						for (var x = 0; x < this.filterWidth; x++) {
							for (var y = 0; y < this.filterHeight; y++) {
								let count = 0;
								let ind = [f * this.inDepth * filterWidth * filterHeight + x + y * filterWidth + d * filterWidth * filterHeight];
								let indR = [
									f * this.inDepth * filterWidth * filterHeight + (x + 1) + y * filterWidth + d * filterWidth * filterHeight,
								];
								let indL = [
									f * this.inDepth * filterWidth * filterHeight + (x - 1) + y * filterWidth + d * filterWidth * filterHeight,
								];
								let indD = [
									f * this.inDepth * filterWidth * filterHeight + x + (y + 1) * filterWidth + d * filterWidth * filterHeight,
								];
								let indU = [
									f * this.inDepth * filterWidth * filterHeight + x + (y - 1) * filterWidth + d * filterWidth * filterHeight,
								];
								if (x < filterWidth - 1) count += this.filterw[indR];
								if (x > 1) count += this.filterw[indL];
								if (y < filterHeight - 1) count += this.filterw[indD];
								if (y > 1) count += this.filterw[indU];
								newFilterw[ind] += count / 5;
							}
						}
					}
				}
				this.filterw = newFilterw;
			}
			if (this.useBias) {
				for (var i = 0; i < this.b.length; i++) {
					this.b[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
			} else {
				this.b.fill(0);
			}
			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//DONT CHANGE THESE OR BIG BREAKY BREAKY!

			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
			this.wIHID = this.inWidth * this.inHeight * this.inDepth;
			this.fWIH = this.filterWidth * this.filterHeight;
			this.fWIHID = this.inDepth * this.filterHeight * this.filterWidth;
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		outSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		inSizeDimensions() {
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.filters,
			];
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

			//-------------Beginning of monstrosity-----------------
			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * this.stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * this.stride;
						for (var h = 0; h < this.inDepth; h++) {
							const hWIH = h * this.wIH + ba;
							const hFWIH = h * this.fWIH + iFWIHID;
							for (var j = 0; j < this.filterHeight; j++) {
								const jGAIWBA = (j + ga) * this.inWidth + hWIH;
								const jFWHFWIH = j * this.filterWidth + hFWIH;
								for (var k = 0; k < this.filterWidth; k++) {
									this.outData[k + jGAIWBA] += this.inData[odi] * this.filterw[k + jFWHFWIH];
								}
							}
						}
					}
				}
			}
			//-------------End of monstrosity-----------------

			for (var i = 0; i < this.outData.length; i++) {
				this.outData[i] += this.b[i];
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			this.costs.fill(0); //reset the costs

			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * this.stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * this.stride;
						for (var h = 0; h < this.inDepth; h++) {
							const hWIH = h * this.wIH;
							const hFWIH = h * this.fWIH + iFWIHID;
							for (var j = 0; j < this.filterHeight; j++) {
								const jGAIWBA = (j + ga) * this.inWidth + hWIH + ba;
								const jFWHFWIH = j * this.filterWidth + hFWIH;
								for (var k = 0; k < this.filterWidth; k++) {
									this.costs[odi] += this.filterw[k + jFWHFWIH] * err[k + jGAIWBA];
									this.accessed[odi]++;
									this.filterws[k + jFWHFWIH] += this.inData[odi] * err[k + jGAIWBA];
								}
							}
						}
					}
				}
			}

			for (var i = 0; i < this.outData.length; i++) {
				this.bs[i] += err[i];
			}
			if (DeconvLayer.averageOutCosts) {
				for (var i = 0; i < this.inSize(); i++) {
					this.costs[i] = this.costs[i] / (this.accessed[i] > 0 ? this.accessed[i] : 1);
					this.accessed[i] = 0;
				}
			}
		}

		getParamsAndGrads(forUpdate = true) {
			if (forUpdate) {
				if (DeconvLayer.averageOutGrads) {
					for (var i = 0; i < this.filterws.length; i++) {
						this.filterws[i] /= this.filters;
					}
				}
			}
			if (this.useBias) {
				return [this.filterw, this.filterws, this.b, this.bs];
			} else {
				return [this.filterw, this.filterws];
			}
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "filterws" ||
					key == "filterbs" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "accessed" ||
					key == "pl" ||
					key == "bs" ||
					key == "ws" ||
					key == "hMFHPO" ||
					key == "wMFWPO" ||
					key == "hMFWMF" ||
					key == "wIH" ||
					key == "wIHID" ||
					key == "fWIH" ||
					key == "fWIHID" ||
					key == (this.useBias ? null : "b") //maybe bad
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, filters = 3, stride = 1,
			let saveObject = JSON.parse(json);
			let layer = new DeconvLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.filters,
				saveObject.stride,
				saveObject.useBias
			);
			for (var i = 0; i < layer.filterw.length; i++) {
				layer.filterw[i] = saveObject.filterw[i];
			}
			if (saveObject.useBias) {
				for (var i = 0; i < layer.b.length; i++) {
					layer.b[i] = saveObject.b[i];
				}
			}
			return layer;
		}
	}

	Ment.DeconvLayer = DeconvLayer;
	Ment.DeConvLayer = DeconvLayer;
	Ment.Deconv = DeconvLayer;
	Ment.DeConv = DeconvLayer;

	//only uncomment if you know what your doing
}

{
	class DepaddingLayer {
		constructor(outDim, pad) {
			if (outDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let outWidth = outDim[0];
			let outHeight = outDim[1];
			let outDepth = outDim[2];

			pad = pad || 2;
			this.outData = new Float32Array(outWidth * outHeight * outDepth);
			this.inData = new Float32Array((outWidth + pad * 2) * (outHeight + pad * 2) * outDepth);
			this.pad = pad;
			this.outWidth = outWidth;
			this.outHeight = outHeight;
			this.outDepth = outDepth;
			this.inData.fill(0);
			this.outData.fill(0);
			this.costs = new Float32Array(this.inData.length);
			this.costs.fill(0);
		}

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw inputError(this, inData);
				}
				//Fun fact: the fastest way to copy an array in javascript is to use a For Loop
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var i = 0; i < this.outDepth; i++) {
				for (var j = 0; j < this.outHeight; j++) {
					for (var h = 0; h < this.outWidth; h++) {
						let prop = i * this.outHeight * this.outWidth + j * this.outWidth + h;
						this.outData[prop] =
							this.inData[
								(j + 1) * this.pad * 2 +
									-this.pad +
									this.pad * (this.outWidth + this.pad * 2) +
									prop +
									i * ((this.outWidth + this.pad * 2) * this.pad) * 2 +
									i * (this.outHeight * this.pad) * 2
							];
					}
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			this.costs.fill(0);
			for (var i = 0; i < this.outDepth; i++) {
				for (var j = 0; j < this.outHeight; j++) {
					for (var h = 0; h < this.outWidth; h++) {
						let prop = i * this.outHeight * this.outWidth + j * this.outWidth + h;
						this.costs[
							(j + 1) * this.pad * 2 +
								-this.pad +
								this.pad * (this.outWidth + this.pad * 2) +
								prop +
								i * ((this.outWidth + this.pad * 2) * this.pad) * 2 +
								i * (this.outHeight * this.pad) * 2
						] = err[prop];
					}
				}
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		outSizeDimensions() {
			return [this.outWidth, this.outHeight, this.outDepth];
		}

		inSizeDimensions() {
			return [this.outWidth + this.pad * 2, this.outHeight + this.pad * 2, this.outDepth];
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "accessed"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new DepaddingLayer([saveObject.outWidth, saveObject.outHeight, saveObject.outDepth], saveObject.pad);
			return layer;
		}
	}

	Ment.DePaddingLayer = DepaddingLayer;
	Ment.DepaddingLayer = DepaddingLayer;
	Ment.DepadLayer = DepaddingLayer;
	Ment.Depadding = DepaddingLayer;
	Ment.Depad = DepaddingLayer;
}

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
			forward(inData::array)
			backward(ExpectedOutput::array) -- returns loss of the layer (Expected - outData)^2
			updateParams(optimizer::string) // optimizer examples: "SGD" "adam" "adagrad" "adadelta"
			save() -- returns json string of layer

		REQUIRED STATIC FUNCTIONS:
			load(json) -- returns a layer  EX: FCLayer.load(layer.save()); //makes a copy of "layer"
			
		REQUIRED FEILDS:
			inData:Float32Array
			outData:Float32Array
			costs:Float32Array
			gpuEnabled:Boolean
			
			----Everything above is the bare minimum for your layer to work at all--------------
			
		NOT REQUIRED FUNCTIONS:
			inSizeDimensions() -- returns array of dimensions i.e. [28,28,3]
			outSizeDimensions()
			
		FEILDS SET BY NET OBJECT:
			lr -- aka learning rate 
			nextLayer -- a reference to the layer after
			previousLayer -- a reference to preceding layer		

		FUNCTIONS FOR EVOLOUTION BASED LEARNING TO WORK:
			mutate(mutationRate::float, mutationIntensity::float)

		FUNCTIONS FOR GPU SUPPORT TO WORK:
			enableGPUMode() -- this should at least make "this.gpuEnabled" true.


			If you have all this your layer should work inside a real net!

		

	*/

	class FCLayer {
		constructor(inSize, outSize, useBias) {
			this.useBias = useBias == undefined ? true : useBias;
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
				if (this.useBias == false) {
					this.b[j] = 0;
				} else {
					this.b[j] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
				this.bs[j] = 0;
			} ///---------adding random biases
		}

		forward(inData) {
			// console.log(inData);
			if (inData) {
				if (inData.length != this.inSize()) {
					throw inputError(this, inData);
				}
				//Fun fact: the fastest way to copy an array in javascript is to use a For Loop
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}
			// console.log(this.inData);

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = 0; //reset outData activation
				for (var j = 0; j < this.inSize(); j++) {
					this.outData[h] += this.inData[j] * this.w[h + j * this.outSize()]; // the dirty deed
				}

				this.outData[h] += this.b[h];
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			for (var i = 0; i < this.inSize(); i++) {
				this.costs[i] = 0;
				for (var j = 0; j < this.outSize(); j++) {
					//activation times error = change to the weight.
					this.ws[j + i * this.outSize()] += this.inData[i] * err[j] * 2;
					this.costs[i] += this.w[j + i * this.outSize()] * err[j] * 2;
				}
			}

			for (var j = 0; j < this.outSize(); j++) {
				this.bs[j] += err[j]; //bias grad so easy
			}
			//finish averaging the costs : required code
			for (var i = 0; i < this.inSize(); i++) {
				this.costs[i] = this.costs[i] / this.outSize();
			}
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
					key == "outData" ||
					key == "costs" ||
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
			layer.lr = saveObject.lr;
			return layer;
		}
	}

	Ment.FCLayer = FCLayer;
	Ment.FC = FCLayer;
}

{
	class IdentityLayer {
		//this layer outputs the inputs with no changes
		constructor(size) {
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(size); //the inData
			this.outData = new Float32Array(size); //will be init when "connect" is called.
			this.costs = new Float32Array(size); //costs for each neuron
			this.pl; //reference to previous layer
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
				this.outData[h] = this.inData[h];
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			for (var j = 0; j < this.outData.length; j++) {
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

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "pl" ||
					key == "outData" ||
					key == "costs" ||
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
			let layer = new IdentityLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.IdentityLayer = IdentityLayer;
	Ment.Identity = IdentityLayer;
	Ment.Input = IdentityLayer;
	Ment.DummyLayer = IdentityLayer;
	Ment.Dummy = IdentityLayer;
	Ment.InputLayer = IdentityLayer;
	Ment.OutputLayer = IdentityLayer;
	Ment.Output = IdentityLayer;
}

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
			this.inData = new Float32Array(size); //the inData
			this.outData = new Float32Array(size + inputSize); //will be init when "connect" is called.
			this.costs = new Float32Array(size); //costs for each neuron
			this.currentInput = new Float32Array(inputSize);
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
			// 	this.inData = new Float32Array(layer.outSize());
			// 	this.outData = new Float32Array(layer.outSize());
			// 	this.costs = new Float32Array(layer.outSize());
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

{
	class LeakyReluLayer extends Ment.ActivationBase {
		static leakySlope = 0.01;

		constructor(size) {
			super(size);
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
				this.outData[h] = this.inData[h] > 0 ? this.inData[h] : this.inData[h] * LeakyReluLayer.leakySlope;
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			for (var j = 0; j < this.outSize(); j++) {
				if (this.outData[j] >= 0) {
					this.costs[j] = err[j];
				} else {
					this.costs[j] = err[j] * LeakyReluLayer.leakySlope;
				}
			}
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new LeakyReluLayer(saveObject.savedSize);
			return layer;
		}
	}
	Ment.LeakyReluLayer = LeakyReluLayer;
	Ment.LeakyRelu = LeakyReluLayer;
	Ment.LRelu = LeakyReluLayer;
}

{
	class MaxPoolLayer {
		constructor(inDim, filterDim, stride = 1) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			if (filterDim.length != 2) {
				throw (
					this.constructor.name +
					" parameter error: Missing filter dimensions parameter. \n" +
					"First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)"
				);
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];

			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.outData = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.inDepth
			);
			this.inData = new Float32Array(inWidth * inHeight * inDepth);
			this.costs = new Float32Array(inWidth * inHeight * inDepth);
			this.maxIndexes = new Float32Array(this.outData.length);
			this.accessed = new Float32Array(this.costs.length).fill(1);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Max Pool layer error: Pooling size (width / height) cannot be bigger than the inputs corresponding (width/height)";
			}

			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//Change these for broken code!
			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.inDepth,
			];
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

			for (var g = 0; g < this.hMFHPO; g++) {
				const ga = g * this.stride;
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					const ba = b * this.stride;
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						const hWIH = h * this.wIH + ba;
						let max = this.inData[ga * this.inWidth + hWIH];
						for (var j = 0; j < this.filterHeight; j++) {
							const jGAIWBA = (j + ga) * this.inWidth + hWIH;
							for (var k = 0; k < this.filterWidth; k++) {
								if (this.inData[k + jGAIWBA] > max) {
									max = this.inData[k + jGAIWBA];
									this.maxIndexes[odi] = k + jGAIWBA;
								}
							}
						}
						this.outData[odi] = max;
					}
				}
			}
		}

		backward(err) {
			this.costs.fill(0);
			if (!err) {
				err = this.nextLayer.costs;
			}
			for (var g = 0; g < this.hMFHPO; g++) {
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						this.costs[this.maxIndexes[odi]] += err[odi];
						this.accessed[this.maxIndexes[odi]]++;
					}
				}
			}
			for (var i = 0; i < this.accessed.length; i++) {
				if (this.accessed[i] != 0) {
					this.costs[i] /= this.accessed[i]; //Average it yeah!!!
					this.accessed[i] = 0;
				}
			}
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, stride = 1,
			let saveObject = JSON.parse(json);
			let layer = new MaxPoolLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.stride
			);
			return layer;
		}
	}

	Ment.MaxPoolLayer = MaxPoolLayer;
	Ment.MaxPoolingLayer = MaxPoolLayer;
	Ment.MaxPool = MaxPoolLayer;
}

{
	class MinPoolLayer {
		constructor(inDim, filterDim, stride = 1) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			if (filterDim.length != 2) {
				throw (
					this.constructor.name +
					" parameter error: Missing filter dimensions parameter. \n" +
					"First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)"
				);
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];

			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.outData = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.inDepth
			);
			this.inData = new Float32Array(inWidth * inHeight * inDepth);
			this.costs = new Float32Array(inWidth * inHeight * inDepth);
			this.minIndexes = new Float32Array(this.outData.length);
			this.accessed = new Float32Array(this.costs.length).fill(1);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Min Pool layer error: Pooling size (width / height) cannot be bigger than the inputs corresponding (width/height)";
			}

			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//Change these for broken code!
			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.inDepth,
			];
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

			for (var g = 0; g < this.hMFHPO; g++) {
				const ga = g * this.stride;
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					const ba = b * this.stride;
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						const hWIH = h * this.wIH + ba;
						let min = this.inData[ga * this.inWidth + hWIH];
						for (var j = 0; j < this.filterHeight; j++) {
							const jGAIWBA = (j + ga) * this.inWidth + hWIH;
							for (var k = 0; k < this.filterWidth; k++) {
								if (this.inData[k + jGAIWBA] < min) {
									min = this.inData[k + jGAIWBA];
									this.minIndexes[odi] = k + jGAIWBA;
								}
							}
						}
						this.outData[odi] = min;
					}
				}
			}
		}

		backward(err) {
			this.costs.fill(0);
			if (!err) {
				err = this.nextLayer.costs;
			}
			for (var g = 0; g < this.hMFHPO; g++) {
				const gWMFWPO = g * this.wMFWPO;
				for (var b = 0; b < this.wMFWPO; b++) {
					for (var h = 0; h < this.inDepth; h++) {
						const odi = b + gWMFWPO + h * this.hMFWMF;
						this.costs[this.minIndexes[odi]] += err[odi];
						this.accessed[this.minIndexes[odi]]++;
					}
				}
			}
			for (var i = 0; i < this.accessed.length; i++) {
				if (this.accessed[i] != 0) {
					this.costs[i] /= this.accessed[i]; //Average it yeah!!!
					this.accessed[i] = 0;
				}
			}
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, stride = 1,
			let saveObject = JSON.parse(json);
			let layer = new MinPoolLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.stride
			);
			return layer;
		}
	}

	Ment.MinPoolLayer = MinPoolLayer;
	Ment.MinPoolingLayer = MinPoolLayer;
	Ment.MinPool = MinPoolLayer;
}

{
	class PaddingLayer {
		constructor(inDim, pad, padwith) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			pad = pad || 2;
			padwith = padwith || 0;
			this.inData = new Float32Array(inWidth * inHeight * inDepth);
			this.outData = new Float32Array((inWidth + pad * 2) * (inHeight + pad * 2) * inDepth);
			this.pad = pad;
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.padwith = padwith;
			this.outData.fill(padwith);
			this.inData.fill(0);
			this.costs = new Float32Array(this.inData.length);
			this.costs.fill(0);
		}

		// const handler = {
		// 	get: function (obj, prop) {
		// 		prop = Math.floor((prop % (inHeight * inWidth)) / inHeight + 1) * padding * 2 + -padding + padding * (inWidth + padding * 2) + parseFloat(prop);
		// 		return obj[prop];
		// 	},
		// 	set: function (obj, prop, value) {
		// 		prop = Math.floor((prop % (inHeight * inWidth)) / inHeight + 1) * padding * 2 + -padding + padding * (inWidth + padding * 2) + parseFloat(prop);
		// 		obj[prop] = value;
		// 	},
		// };

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw inputError(this, inData);
				}
				//Fun fact: the fastest way to copy an array in javascript is to use a For Loop
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}
			this.outData.fill(this.padwith);

			for (var i = 0; i < this.inDepth; i++) {
				for (var j = 0; j < this.inHeight; j++) {
					for (var h = 0; h < this.inWidth; h++) {
						let prop = i * this.inHeight * this.inWidth + j * this.inWidth + h;
						this.outData[
							(j + 1) * this.pad * 2 +
								-this.pad +
								this.pad * (this.inWidth + this.pad * 2) +
								prop +
								i * ((this.inWidth + this.pad * 2) * this.pad) * 2 +
								i * (this.inHeight * this.pad) * 2
						] = this.inData[prop];
					}
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			this.costs.fill(0);

			for (var i = 0; i < this.inDepth; i++) {
				for (var j = 0; j < this.inHeight; j++) {
					for (var h = 0; h < this.inWidth; h++) {
						let prop = i * this.inHeight * this.inWidth + j * this.inWidth + h;
						this.costs[prop] =
							err[
								(j + 1) * this.pad * 2 +
									-this.pad +
									this.pad * (this.inWidth + this.pad * 2) +
									prop +
									i * ((this.inWidth + this.pad * 2) * this.pad) * 2 +
									i * (this.inHeight * this.pad) * 2
							];
					}
				}
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [this.inWidth + this.pad * 2, this.inHeight + this.pad * 2, this.inDepth];
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new PaddingLayer(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				saveObject.pad,
				saveObject.padwith
			);
			return layer;
		}
	}

	Ment.PaddingLayer = PaddingLayer;
	Ment.PadLayer = PaddingLayer;
	Ment.Padding = PaddingLayer;
	Ment.Pad = PaddingLayer;
}

{
	class RecEmitterLayer {
		//Glorified Identity layer, only difference it has an ID and reference to the receiver with same ID
		constructor(id) {
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(0); //the inData
			this.outData = new Float32Array(0); //will be init when "connect" is called.
			this.costs = new Float32Array(0); //costs for each neuron
			this.receiver; // a reference to the receiver layer so we can skip layers
			//this will be set by the receiver  when the net is initialized
			this.savedOutData;
			this.pl = undefined; //the previous layer except interally its called "pl" so i can set getters and setters for the value "previousLayer"
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			if (layer.constructor.name == "RecReceiverLayer" && layer.id == this.id) {
				throw "You can't put a RecReceiver right before its corressponding RecEmitter. (because it doesnt know the output size) (try adding a dummy layer inbetween and giving it a size)";
			}
			this.inData = new Float32Array(layer.outSize());
			this.costs = new Float32Array(layer.outSize());
			this.pl = layer;

			this.outData = new Float32Array(layer.outSize());
			this.savedOutData = new Float32Array(layer.outSize());
			this.savedOutData.fill(0);
		}

		forward(inData) {
			//first save what was last outputted for the receiver
			if (this.where == "behind") {
				//this layer is to the left of receiver
				for (var i = 0; i < this.outSize(); i++) {
					this.savedOutData[i] = this.outData[i];
				}
			}
			if (inData) {
				if (inData.length != this.inSize()) {
					throw Ment.inputError(this, inData);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var h = 0; h < this.outSize(); h++) {
				//the outData of this layer is the same object referenced in the inData of the Receiver layer
				this.outData[h] = this.inData[h];
			}
			if (this.where == "in front") {
				//this layer is to the right
				for (var i = 0; i < this.outSize(); i++) {
					this.savedOutData[i] = this.outData[i];
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			for (var j = 0; j < this.outData.length; j++) {
				this.costs[j] = (err[j] + this.costsFromReceiver[j]) / 2;
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

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "receiver" ||
					key == "pl" ||
					key == "inData" ||
					key == "bs" ||
					key == "ws" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "emitter" ||
					key == "costsFromReceiver"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			//todo check if this is needed
			delete this.savedInSize;
			delete this.savedOutSize;

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new RecEmitterLayer(saveObject.id);
			return layer;
		}
	}

	Ment.RecEmitterLayer = RecEmitterLayer;
	Ment.RecEmitter = RecEmitterLayer;
	Ment.RecE = RecEmitterLayer;
}

{
	class RecReceiverLayer {
		//Recurrent Receiver
		constructor(id, mode = "concat") {
			//mode can be concat or add or average
			this.mode = mode;
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.inData; //the inData
			this.outData; //will be init when "onConnect" is called.
			this.costs; //costs for each neuron
			this.emitter;
			this.inDataFromEmitter;
			this.costsForEmitter;
			this.savedCostsForEmitter; //costs one step behind
			this.pl; // holds a reference to previous layer
			this.nl; //next layer
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inData = new Float32Array(layer.outSize());
			this.costs = new Float32Array(layer.outSize());
			this.pl = layer;
			//time to find this layers soulmate
			let found = false;
			let currentLayer = layer; //start at this layer go back until find a reciever with the same ID
			while (!found) {
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.RecEmitter) {
					found = true;
				} else {
					currentLayer = currentLayer.previousLayer;
					if (currentLayer == undefined) {
						if (this.checkedFront) {
							throw "COULD NOT FIND MATCHING REC EMITTER FOR REC RECEIVER LAYERR";
						}
						this.checkedBehind = true;
						return; //give up for now
					}
				}
			}
			this.emitter = currentLayer;
			this.emitter.where = "behind";

			this.inDataFromEmitter = this.emitter.savedOutData;
			currentLayer.receiver = this; //so they can find each other again :)
			if (this.mode == "add" || this.mode == "average") {
				if (this.pl.outSize() != this.emitter.outSize()) {
					throw "emitter size must equal the size of the previous layer of the corresponding receiver layer";
				}
				this.outData = new Float32Array(layer.outSize());
			} else if (this.mode == "concat") {
				this.outData = new Float32Array(layer.outSize() + this.emitter.outSize());
			}
			this.costsForEmitter = new Float32Array(this.emitter.outSize());
			this.savedCostsForEmitter = new Float32Array(this.emitter.outSize());
			this.emitter.costsFromReceiver = this.savedCostsForEmitter;
		}

		get nextLayer() {
			return this.nl;
		}

		set nextLayer(layer) {
			this.nl = layer;
			//time to find this layers soulmate
			let found = false;
			let currentLayer = layer; //start at this layer go back until find a reciever with the same ID
			while (!found) {
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.RecEmitterLayer) {
					found = true;
				} else {
					currentLayer = currentLayer.nextLayer;

					if (currentLayer == undefined) {
						if (this.checkedBehind) {
							throw "COULD NOT FIND MATCHING REC EMITTER FOR REC RECEIVER LAYER";
						}
						this.checkedFront = true;
						return; //give up for now
					}
				}
			}
			this.emitter = currentLayer;
			this.emitter.where = "in front";
			this.inDataFromEmitter = this.emitter.savedOutData;
			currentLayer.receiver = this; //so they can find each other again :)
			if (this.mode == "add" || this.mode == "average") {
				if (layer.outSize() != this.emitter.outSize()) {
					throw "emitter size must equal the size of the previous layer of the corresponding receiver layer";
				}
				this.outData = new Float32Array(this.previousLayer.outSize());
			} else if (this.mode == "concat") {
				this.outData = new Float32Array(this.previousLayer.outSize() + this.emitter.outSize());
			}
			this.costsForEmitter = new Float32Array(this.emitter.outSize());
			this.savedCostsForEmitter = new Float32Array(this.emitter.outSize());
			this.emitter.costsFromReceiver = this.savedCostsForEmitter;
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
			if (this.mode == "concat") {
				for (var h = 0; h < this.inData.length; h++) {
					this.outData[h] = this.inData[h];
				}
				for (var h = this.inData.length; h < this.inData.length + this.inDataFromEmitter.length; h++) {
					this.outData[h] = this.inDataFromEmitter[h - this.inData.length];
				}
			} else if (this.mode == "add") {
				for (var i = 0; i < this.outData.length; i++) {
					this.outData[i] = this.inData[i] + this.inDataFromEmitter[i];
				}
			} else if (this.mode == "average") {
				for (var i = 0; i < this.outData.length; i++) {
					this.outData[i] = (this.inData[i] + this.inDataFromEmitter[i]) / 2;
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			if (this.emitter.where == "behind") {
				for (var i = 0; i < this.inSize(); i++) {
					this.savedCostsForEmitter[i] = this.costsForEmitter[i];
				}
			}
			if (this.mode == "concat") {
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = err[i];
				}
				for (var i = this.inData.length; i < this.inData.length + this.inDataFromEmitter.length; i++) {
					this.costsForEmitter[i - this.inData.length] = err[i];
				}
			} else if (this.mode == "add") {
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = err[i];
					this.costsForEmitter[i] = err[i];
				}
			} else if (this.mode == "average") {
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = err[i] * 2;
					this.costsForEmitter[i] = err[i] * 2;
				}
			}

			if (this.emitter.where == "in front") {
				for (var i = 0; i < this.inSize(); i++) {
					this.savedCostsForEmitter[i] = this.costsForEmitter[i];
				}
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "emitter" ||
					key == "pl" ||
					key == "ws" ||
					key == "bs" ||
					key == "nl" ||
					key == "receiver" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "costsForEmitter" ||
					key == "inDataFromEmitter" ||
					key == "savedCostsForEmitter" ||
					key == "costsFromReceiver" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "savedOutData"
				) {
					return undefined;
				}

				return value;
			});
			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new RecReceiverLayer(saveObject.id, saveObject.mode);
			return layer;
		}
	}

	Ment.RecReceiverLayer = RecReceiverLayer;
	Ment.RecReceiver = RecReceiverLayer;
	Ment.RecR = RecReceiverLayer;
}

{
	class ReluLayer extends Ment.ActivationBase {
		constructor(size) {
			super(size);
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
				this.outData[h] = this.inData[h] > 0 ? this.inData[h] : 0;
			}
			//Oh the misery
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			for (var j = 0; j < this.outSize(); j++) {
				if (this.outData[j] >= 0) {
					this.costs[j] = err[j];
				}
			}
		}
		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ReluLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.ReluLayer = ReluLayer;
	Ment.Relu = ReluLayer;
}

{
	class ResEmitterLayer {
		//this layer outputs the inputs with no changes
		constructor(id) {
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.inData = new Float32Array(0); //the inData
			this.outData = new Float32Array(0); //will be init when "connect" is called.
			this.costs = new Float32Array(0); //costs for each neuron
			this.receiver; // a reference to the receiver layer so we can skip layers
			//this will be set by the receiver  when the net is initialized
			this.pl = undefined;
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inData = new Float32Array(layer.outSize());
			this.costs = new Float32Array(layer.outSize());
			this.pl = layer;

			this.outData = new Float32Array(layer.outSize());
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
				//the outData of this layer is the same object referenced in the inData of the Receiver layer
				this.outData[h] = this.inData[h];
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			this.costs.fill(0);

			for (var i = 0; i < this.outData.length; i++) {
				this.costs[i] += err[i];
				this.costs[i] += this.receiver.costsForEmitter[i];
				this.costs[i] /= 2;
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

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "receiver" ||
					key == "pl" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "emitter"
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

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ResEmitterLayer(saveObject.id);
			return layer;
		}
	}

	Ment.ResEmitterLayer = ResEmitterLayer;
	Ment.ResEmitter = ResEmitterLayer;
	Ment.ResE = ResEmitterLayer;
}

{
	class ResReceiverLayer {
		//this layer outputs the inputs with no changes
		constructor(id, mode = "concat") {
			//mode can be concat or add or average
			this.mode = mode;
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.inData; //the inData
			this.outData; //will be init when "onConnect" is called.
			this.costs; //costs for each neuron
			this.emitter;
			this.inDataFromEmitter;
			this.costsForEmitter;
			this.pl; // holds a reference to previous layer
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inData = new Float32Array(layer.outSize());
			this.costs = new Float32Array(layer.outSize());
			this.pl = layer;
			//time to find this layers soulmate
			let found = false;
			let currentLayer = layer; //start at this layer go backward until find a emitter with the same ID
			while (!found) {
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.ResEmitter) {
					found = true;
				} else {
					currentLayer = currentLayer.previousLayer;
					if (currentLayer == undefined) {
						throw "Could not find Matching Emitter Layer for Receiver Layer ID: " + this.id;
					}
				}
			}
			this.emitter = currentLayer; //so they can find each other again :)
			this.inDataFromEmitter = this.emitter.outData;
			currentLayer.receiver = this;
			if (this.mode == "add" || this.mode == "average") {
				if (layer.outSize() != this.emitter.outSize()) {
					throw "emitter size must equal the size of the previous layer of the corresponding receiver layer";
				}
				this.outData = new Float32Array(layer.outSize());
			} else if (this.mode == "concat") {
				this.outData = new Float32Array(layer.outSize() + this.emitter.outSize());
			}
			this.costsForEmitter = new Float32Array(this.emitter.outSize());
		}

		//we dont look in front because residual data only goes forwards.

		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw Ment.inputError(this, inData);
				}
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}
			if (this.mode == "concat") {
				//first the data behind it and then the skip layer data
				for (var h = 0; h < this.inData.length; h++) {
					this.outData[h] = this.inData[h];
				}
				for (var h = this.inData.length; h < this.inData.length + this.inDataFromEmitter.length; h++) {
					this.outData[h] = this.inDataFromEmitter[h - this.inData.length];
				}
			} else if (this.mode == "add") {
				for (var i = 0; i < this.outData.length; i++) {
					this.outData[i] = this.inData[i] + this.inDataFromEmitter[i];
				}
			} else if (this.mode == "average") {
				for (var i = 0; i < this.outData.length; i++) {
					this.outData[i] = (this.inData[i] + this.inDataFromEmitter[i]) / 2;
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}

			if (this.mode == "concat") {
				//fixed a very atrocious error
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = err[i];
				}
				for (var i = this.inData.length; i < this.inData.length + this.inDataFromEmitter.length; i++) {
					this.costsForEmitter[i - this.inData.length] = err[i];
				}
			} else if (this.mode == "add") {
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = err[i] / 2;
					this.costsForEmitter[i] = err[i] / 2;
				}
			} else if (this.mode == "average") {
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = err[i] * 2;
					this.costsForEmitter[i] = err[i] * 2;
				}
			}
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "emitter" ||
					key == "pl" ||
					key == "receiver" ||
					key == "ws" ||
					key == "bs" ||
					key == "nl" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "costsForEmitter" ||
					key == "inDataFromEmitter"
				) {
					return undefined;
				}

				return value;
			});
			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ResReceiverLayer(saveObject.id, saveObject.mode);
			return layer;
		}
	}

	Ment.ResReceiverLayer = ResReceiverLayer;
	Ment.ResReceiver = ResReceiverLayer;
	Ment.ResR = ResReceiverLayer;
}

{
	class SigmoidLayer extends Ment.ActivationBase {
		sigmoid(z) {
			return 1 / (1 + Math.exp(-z));
		}

		sigmoidPrime(z) {
			let ret = Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2);
			if (isNaN(ret)) {
				return 0;
			}
			return ret;
		}

		constructor(size) {
			super(size);
		}

		forward(inData) {
			super.forward(inData, this.sigmoid);
		}

		backward(expected) {
			return super.backward(expected, this.sigmoidPrime);
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new SigmoidLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.SigmoidLayer = SigmoidLayer;
	Ment.Sigmoid = SigmoidLayer;
	Ment.Sig = SigmoidLayer;
}

{
	class SineLayer extends Ment.ActivationBase {
		//worst.. most useless layer ever.
		// :(
		sine(z) {
			return Math.sin(z);
		}

		sinePrime(z) {
			return Math.cos(z);
		}

		constructor(size) {
			super(size);
		}

		forward(inData) {
			return super.forward(inData, this.sine);
		}

		backward(expected) {
			return super.backward(expected, this.sinePrime);
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}
		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new SineLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.SineLayer = SineLayer;
	Ment.Sine = SineLayer;
	Ment.Sin = SineLayer;
}

{
	class TanhLayer extends Ment.ActivationBase {
		tanh(z) {
			return Math.tanh(z);
		}

		tanhPrime(z) {
			return 1 - Math.pow(Math.tanh(z), 2);
		}

		constructor(size) {
			super(size);
		}

		forward(inData) {
			super.forward(inData, this.tanh);
		}

		backward(expected) {
			return super.backward(expected, this.tanhPrime);
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new TanhLayer(saveObject.savedSize);
			return layer;
		}
	}

	Ment.TanhLayer = TanhLayer;
	Ment.Tanh = TanhLayer;
}

{
	class UpscalingLayer {
		constructor(inDim, scale) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.outWidth = inWidth * scale;
			this.outHeight = inHeight * scale;
			this.scale = scale;
			this.nextLayer; //the connected layer
			this.previousLayer; //the previousLayer
			this.inData = new Float32Array(inWidth * inHeight * inDepth); //the inData
			this.outData = new Float32Array(this.outWidth * this.outHeight * inDepth);
			this.costs = new Float32Array(this.inData.length); //costs for each activation in "inData"
		}
		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [this.outWidth, this.outHeight, this.inDepth];
		}
		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw inputError(this, inData);
				}
				//Fun fact: the fastest way to copy an array in javascript is to use a For Loop
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for (var i = 0; i < this.inDepth; i++) {
				for (var h = 0; h < this.inHeight * this.scale; h++) {
					for (var j = 0; j < this.inWidth * this.scale; j++) {
						this.outData[i * this.outHeight * this.outWidth + h * this.outWidth + j] =
							this.inData[
								Math.floor(i) * this.inHeight * this.inWidth +
									Math.floor(h / this.scale) * this.inWidth +
									Math.floor(j / this.scale)
							];
					}
				}
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.costs;
			}
			this.costs.fill(0);

			for (var i = 0; i < this.inDepth; i++) {
				//this can be optimized
				for (var h = 0; h < this.inHeight * this.scale; h++) {
					for (var j = 0; j < this.inWidth * this.scale; j++) {
						let t = err[i * this.outHeight * this.outWidth + h * this.outWidth + j];
						this.costs[
							Math.floor(i) * this.inHeight * this.inWidth +
								Math.floor(h / this.scale) * this.inWidth +
								Math.floor(j / this.scale)
						] += t;
					}
				}
			}
			// for(var i = 0;i<this.costs.length;i++){
			// 	this.costs[i] /= this.scale;
			// }
		}

		inSize() {
			return this.inData.length;
		}

		outSize() {
			return this.outData.length;
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "ws" ||
					key == "bs" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new UpscalingLayer([saveObject.inWidth, saveObject.inHeight, saveObject.inDepth], saveObject.scale);
			return layer;
		}
	}

	Ment.UpscalingLayer = UpscalingLayer;
	Ment.Upscaling = UpscalingLayer;
	Ment.Upscale = UpscalingLayer;
	Ment.UpScale = UpscalingLayer;
	Ment.UpScaling = UpscalingLayer;
	Ment.UpScalingLayer = UpscalingLayer;
}

//Base layer for all activations to inherit from

{
	class ActivationBaseGPU {
		constructor(size) {
			this.nextLayer; //the connected layer
			this.pl; //reference to previous layer
			this.inSize = () => {
				return size;
			};

			this.outSize = () => {
				return size;
			};
		}

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			//will get initialized berore initGPUactivations is run.
			this.inSize = () => {
				return layer.outSize();
			};

			this.outSize = () => {
				return layer.outSize();
			};
			this.pl = layer;
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuBiasName" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "gpuBiasGradsName" ||
					key == "gpuWeightsName" ||
					key == "gpuWeightsGradsName" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			//This is how you delete object properties btw.
			delete this.savedInSize;

			return ret;
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}
	}

	Ment.ActivationBaseGPU = ActivationBaseGPU;
}

{
	class ConvLayerGPU {
		//yeah these averaging things wont do antyhing cuz i havent implemented them yet
		//dont worry the layer will still werk
		//Proggramming this layer right here gave me schizofrinia!!!
		static averageOutCosts = true; //probs
		static averageOutGrads = true; //ehhh
		constructor(inDim, filterDim, filters = 3, stride = 1, bias = true) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			if (filterDim.length != 2) {
				throw (
					this.constructor.name +
					" parameter error: Missing filter dimensions parameter. \n" +
					"First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)"
				);
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];
			this.filters = filters; //the amount of filters
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;

			this.gpuFilterW = Ment.makeid(8); //new Float32Array(filters * inDepth * filterWidth * filterHeight);
			let temp = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			for (var i = 0; i < filters * inDepth * filterWidth * filterHeight; i++) {
				temp[i] = 0.3 * Math.random() * (Math.random() > 0.5 ? -1 : 1); //set random weights
			}
			Ment.webMonkeys.set(this.gpuFilterW, temp);

			this.gpuFilterWGrads = Ment.makeid(8); //new Float32Array(filters * inDepth * filterWidth * filterHeight);
			Ment.webMonkeys.set(this.gpuFilterWGrads, filters * inDepth * filterWidth * filterHeight);

			// this.accessed = new Float32Array(this.inData.length).fill(0); //to average out the costs

			this.gpuBiasName = Ment.makeid(8); //new Float32Array(this.outData.length);
			this.gpuBiasGradsName = Ment.makeid(8); //new Float32Array(this.outData.length);
			this.useBias = bias;
			temp = new Float32Array(
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			if (this.useBias) {
				for (
					var i = 0;
					i < Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters;
					i++
				) {
					temp[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1); // set random bias
				}
			}
			Ment.webMonkeys.set(this.gpuBiasName, temp);
			Ment.webMonkeys.set(
				this.gpuBiasGradsName,
				Math.ceil((inWidth - filterWidth + 1) / stride) * Math.ceil((inHeight - filterHeight + 1) / stride) * this.filters
			);
			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Conv layer error: filters cannot be bigger than the input";
			}

			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//DONT CHANGE THESE OR BIG BREAKY BREAKY!

			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
			this.wIHID = this.inWidth * this.inHeight * this.inDepth;
			this.fWIH = this.filterWidth * this.filterHeight;
			this.fWIHID = this.inDepth * this.filterHeight * this.filterWidth;

			this.gpuAccessedMap = Ment.makeid(8); //a map with the same size of the input times filter each input pixel will store what filter pixels can reach it. Needed cuz gpu is so so dumb unlike cpu

			//initialize it.
			// prettier-ignore
			{
			let tempAccessed = new Float32Array(inWidth * inHeight * filterWidth * filterHeight);

			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * this.stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * this.stride;
						for (var h = 0; h < this.inDepth; h++) {
							const hWIH = h * this.wIH;
							const hFWIH = h * this.fWIH + iFWIHID;
							for (var j = 0; j < this.filterHeight; j++) {
								const jGAIWBA = (j + ga) * this.inWidth + hWIH + ba;
								const jFWHFWIH = j * this.filterWidth + hFWIH;
								for (var k = 0; k < this.filterWidth; k++) {
									let filterInd = j * this.filterWidth + k;
									let indice = (k + jGAIWBA)*this.filterHeight * this.filterWidth + filterInd;
									tempAccessed[indice] = g * this.wMFWPO + b + 1.1;
								}
							}
						}
					}
				}
			}

			Ment.webMonkeys.set(this.gpuAccessedMap, tempAccessed);


		}
		}

		inSize() {
			return this.inWidth * this.inHeight * this.inDepth;
		}

		outSize() {
			return (
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride) *
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride) *
				this.filters
			);
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.filters,
			];
		}

		forward() {
			Ment.webMonkeys.work(
				this.filters * this.hMFHPO * this.wMFWPO,
				`
float act = 0.0;
int ifromi = int(i / ${this.hMFHPO * this.wMFWPO});
int gfromi = int((i - ifromi * ${this.hMFHPO * this.wMFWPO}) / ${this.wMFWPO});
int bfromi = int((i - ifromi * ${this.hMFHPO * this.wMFWPO} - gfromi * ${this.wMFWPO}));


int iHMFWMF = ifromi * ${this.hMFWMF};
int iFWIHID = ifromi * ${this.fWIHID};
int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};

for (int h = 0; h < ${this.inDepth}; h++) {
	int hWIH = h * ${this.wIH} + ba;
	int hFWIH = h * ${this.fWIH} + iFWIHID;

	for (int j = 0; j < ${this.filterHeight}; j++) {
		int jGAIWBA = (j + ga) * ${this.inWidth} + hWIH;
		int jFWHFWIH = j * ${this.filterWidth} + hFWIH;
		for (int k = 0; k < ${this.filterWidth}; k++) {
			act += ${this.gpuInDataName}(k + jGAIWBA + ${this.gpuInDataStartIndex}) * ${this.gpuFilterW}(k + jFWHFWIH);
		}
	}
}
act += ${this.gpuBiasName}(odi);

;

${this.gpuOutDataName}(odi + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}
		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			//v will be height c will be width h will be depth
			//this took forever to think of
			Ment.webMonkeys.work(
				this.inWidth * this.inHeight * this.inDepth,
				`
float act = 0.0;
int remaining = i;
int hfromi = int(remaining / (${this.inWidth * this.inHeight}));
remaining = remaining - (hfromi * ${this.inWidth * this.inHeight});
int vfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (vfromi * ${this.inWidth});
int cfromi = remaining;
int cdi = i;
int acc = ((i - hfromi * ${this.inWidth * this.inHeight}) * ${this.filterHeight * this.filterWidth});

for (int jfromi = 0; jfromi < ${this.filterHeight}; jfromi++) {
for (int kfromi = 0; kfromi < ${this.filterWidth}; kfromi++) {
if(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi) > 0.5){

int amnt = int(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi)) - 1;
int gfromi = int(amnt / ${this.wMFWPO});
int bfromi = amnt - gfromi * ${this.wMFWPO};
for (int ifromi = 0; ifromi < ${this.filters}; ifromi++) {
int iHMFWMF = ifromi * ${this.hMFWMF};
int iFWIHID = ifromi * ${this.fWIHID};

int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};
int hWIH = hfromi * ${this.wIH};
int hFWIH = hfromi * ${this.fWIH} + iFWIHID;
int jFWHFWIH = jfromi * ${this.filterWidth} + hFWIH;

act += ${this.gpuFilterW}(kfromi + jFWHFWIH) * ${this.gpuErrorArrayName}(odi);

}



}
}
};

;

${this.gpuCostsArrayName}(cdi) := act;
`
			);

			Ment.webMonkeys.work(
				this.filters * this.inDepth * this.filterWidth * this.filterHeight,
				`
				int remaining = i;
				int ifromi = int(remaining / (${this.inDepth * this.filterHeight * this.filterWidth}));
remaining = remaining - (ifromi * ${this.inDepth * this.filterHeight * this.filterWidth});
int hfromi = int(remaining / (${this.filterHeight * this.filterWidth}));
remaining = remaining - (hfromi * ${this.filterHeight * this.filterWidth});
int jfromi = int(remaining / (${this.filterWidth}));
remaining = remaining - (jfromi * ${this.filterWidth});
int kfromi = remaining;


int cdi = i;

int iFWIHID = ifromi * ${this.fWIHID};
int hFWIH = hfromi * ${this.fWIH} + iFWIHID;
int jFWHFWIH = jfromi * ${this.filterWidth} + hFWIH;
float act = ${this.gpuFilterWGrads}(kfromi + jFWHFWIH);
for (int vfromi = 0; vfromi < ${this.inHeight}; vfromi++) {
for (int cfromi = 0; cfromi < ${this.inWidth}; cfromi++) {
int acc = (((vfromi * ${this.inWidth} + cfromi)) * ${this.filterHeight * this.filterWidth});
if(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi) > 0.5){

int amnt = int(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi)) - 1;
int gfromi = int(amnt / ${this.wMFWPO});
int bfromi = amnt - gfromi * ${this.wMFWPO};
int iHMFWMF = ifromi * ${this.hMFWMF};

int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};
int hWIH = hfromi * ${this.wIH};
int jGAIWBA = (jfromi + ga) * ${this.inWidth} + hWIH + ba;

act += ${this.gpuInDataName}(kfromi + jGAIWBA + ${this.gpuInDataStartIndex}) * ${this.gpuErrorArrayName}(odi);





}
}
};

;

${this.gpuFilterWGrads}(kfromi + jFWHFWIH) := act;
`
			);

			if (this.useBias) {
				Ment.webMonkeys.work(
					this.outSize(),
					`
float act = ${this.gpuBiasGradsName}(i) + ${this.gpuErrorArrayName}(i);

;

${this.gpuBiasGradsName}(i) := act;
`
				);
			}
		}

		getParamsAndGrads(forUpdate = true) {
			if (this.useBias) {
				return [this.gpuFilterW, this.gpuFilterWGrads, this.gpuBiasName, this.gpuBiasGradsName];
			} else {
				return [this.gpuFilterW, this.gpuFilterWGrads];
			}
		}

		save() {
			this.filterw = Ment.webMonkeys.get(this.gpuFilterW);
			this.b = Ment.webMonkeys.get(this.gpuBiasName);
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "filterws" ||
					key == "gpuFilterWGrads" ||
					key == "gpuBiasGradsName" ||
					key == "gpuAccessedMap" ||
					key == "filterbs" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "accessed" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "bs" ||
					key == "ws" ||
					key == "hMFHPO" ||
					key == "wMFWPO" ||
					key == "hMFWMF" ||
					key == "wIH" ||
					key == "wIHID" ||
					key == "fWIH" ||
					key == "fWIHID" ||
					key == (this.useBias ? null : "b")
				) {
					return undefined;
				}

				return value;
			});

			delete this.filterw;
			delete this.b;
			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new ConvLayerGPU(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.filters,
				saveObject.stride,
				saveObject.useBias
			);
			let temp = new Float32Array(layer.filters * layer.inDepth * layer.filterWidth * layer.filterHeight);
			for (var i = 0; i < layer.filters * layer.inDepth * layer.filterWidth * layer.filterHeight; i++) {
				temp[i] = saveObject.filterw[i];
			}
			Ment.webMonkeys.set(layer.gpuFilterW, temp);
			if (layer.useBias) {
				temp = new Float32Array(layer.outSize());
				for (var i = 0; i < layer.outSize(); i++) {
					temp[i] = saveObject.b[i];
				}
				Ment.webMonkeys.set(layer.gpuBiasName, temp);
			}
			return layer;
		}
	}

	Ment.ConvLayerGPU = ConvLayerGPU;
	Ment.ConvGPU = ConvLayerGPU;
}

{
	class DeconvLayerGPU {
		static averageOutCosts = true;
		static averageOutGrads = true;
		constructor(inDim, filterDim, filters = 3, stride = 1, useBias = true) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			if (filterDim.length != 2) {
				throw (
					this.constructor.name +
					" parameter error: Missing filter dimensions parameter. \n" +
					"First parameter in layer must be an 2 length array, width height. (filter depth is always the input depth)"
				);
			}
			let filterWidth = filterDim[0];
			let filterHeight = filterDim[1];
			this.useBias = useBias;
			this.filters = filters; //the amount of filters
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.filterWidth = filterWidth;
			this.filterHeight = filterHeight;
			this.stride = stride;
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;

			if (this.filterWidth > inWidth || this.filterHeight > inHeight) {
				throw "Conv layer error: filters cannot be bigger than the input";
			}
			//init random weights
			this.gpuFilterW = Ment.makeid(8); //new Float32Array(filters * inDepth * filterWidth * filterHeight);
			let temp = new Float32Array(filters * inDepth * filterWidth * filterHeight);
			for (var i = 0; i < filters * inDepth * filterWidth * filterHeight; i++) {
				temp[i] = 0.3 * Math.random() * (Math.random() > 0.5 ? -1 : 1); //set random weights
			}
			Ment.webMonkeys.set(this.gpuFilterW, temp);

			this.gpuFilterWGrads = Ment.makeid(8); //new Float32Array(filters * inDepth * filterWidth * filterHeight);
			Ment.webMonkeys.set(this.gpuFilterWGrads, filters * inDepth * filterWidth * filterHeight);

			// this.accessed = new Float32Array(this.inData.length).fill(0); //to average out the costs

			this.gpuBiasName = Ment.makeid(8); //new Float32Array(this.outData.length);
			this.gpuBiasGradsName = Ment.makeid(8);
			temp = new Float32Array(this.inHeight * this.inWidth * this.inDepth);
			if (this.useBias) {
				for (var i = 0; i < this.inHeight * this.inWidth * this.inDepth; i++) {
					temp[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1); // set random bias
				}
			}
			Ment.webMonkeys.set(this.gpuBiasName, temp);
			Ment.webMonkeys.set(this.gpuBiasGradsName, this.inHeight * this.inWidth * this.inDepth);

			//Everything below here is precalculated constants used in forward/backward
			//to optimize this and make sure we are as effeiciant as possible.
			//DONT CHANGE THESE OR BIG BREAKY BREAKY!

			this.hMFHPO = Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride);
			this.wMFWPO = Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride);
			this.hMFWMF = this.hMFHPO * this.wMFWPO;
			this.wIH = this.inWidth * this.inHeight;
			this.wIHID = this.inWidth * this.inHeight * this.inDepth;
			this.fWIH = this.filterWidth * this.filterHeight;
			this.fWIHID = this.inDepth * this.filterHeight * this.filterWidth;

			this.gpuAccessedMap = Ment.makeid(8); //a map with the same size of the input times filter each input pixel will store what filter pixels can reach it. Needed cuz gpu is so so dumb unlike cpu

			//initialize it.
			Ment.webMonkeys.set(this.gpuAccessedMap, inWidth * inHeight * filterWidth * filterHeight);
			// prettier-ignore
			{
			let tempAccessed = new Float32Array(inWidth * inHeight * filterWidth * filterHeight);

			for (var i = 0; i < this.filters; i++) {
				const iHMFWMF = i * this.hMFWMF;
				const iFWIHID = i * this.fWIHID;
				for (var g = 0; g < this.hMFHPO; g++) {
					const ga = g * this.stride;
					const gWMFWPO = g * this.wMFWPO;
					for (var b = 0; b < this.wMFWPO; b++) {
						const odi = b + gWMFWPO + iHMFWMF;
						const ba = b * this.stride;
						for (var h = 0; h < this.inDepth; h++) {
							const hWIH = h * this.wIH;
							const hFWIH = h * this.fWIH + iFWIHID;
							for (var j = 0; j < this.filterHeight; j++) {
								const jGAIWBA = (j + ga) * this.inWidth + hWIH + ba;
								const jFWHFWIH = j * this.filterWidth + hFWIH;
								for (var k = 0; k < this.filterWidth; k++) {
									let filterInd = j * this.filterWidth + k;
									let indice = (k + jGAIWBA)*this.filterHeight * this.filterWidth + filterInd;
									tempAccessed[indice] = g * this.wMFWPO + b + 1.1;
								}
							}
						}
					}
				}
			}

			Ment.webMonkeys.set(this.gpuAccessedMap, tempAccessed);


		}
		}

		inSize() {
			return (
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride) *
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride) *
				this.filters
			);
		}

		outSize() {
			return this.inWidth * this.inHeight * this.inDepth;
		}

		outSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		inSizeDimensions() {
			return [
				Math.ceil((this.inWidth - this.filterWidth + 1) / this.stride),
				Math.ceil((this.inHeight - this.filterHeight + 1) / this.stride),
				this.filters,
			];
		}

		forward() {
			Ment.webMonkeys.work(
				this.inWidth * this.inHeight * this.inDepth,
				`
float act = 0.0;
int remaining = i;
int hfromi = int(remaining / (${this.inWidth * this.inHeight}));
remaining = remaining - (hfromi * ${this.inWidth * this.inHeight});
int vfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (vfromi * ${this.inWidth});
int cfromi = remaining;
int cdi = i;
int acc = ((i - hfromi * ${this.inWidth * this.inHeight}) * ${this.filterHeight * this.filterWidth});

for (int jfromi = 0; jfromi < ${this.filterHeight}; jfromi++) {
for (int kfromi = 0; kfromi < ${this.filterWidth}; kfromi++) {
if(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi) > 0.5){

int amnt = int(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi)) - 1;
int gfromi = int(amnt / ${this.wMFWPO});
int bfromi = amnt - gfromi * ${this.wMFWPO};
for (int ifromi = 0; ifromi < ${this.filters}; ifromi++) {
int iHMFWMF = ifromi * ${this.hMFWMF};
int iFWIHID = ifromi * ${this.fWIHID};

int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};
int hWIH = hfromi * ${this.wIH};
int hFWIH = hfromi * ${this.fWIH} + iFWIHID;
int jFWHFWIH = jfromi * ${this.filterWidth} + hFWIH;

act += ${this.gpuFilterW}(kfromi + jFWHFWIH) * ${this.gpuInDataName}(odi + ${this.gpuInDataStartIndex});

}



}
}
};

act += ${this.gpuBiasName}(i);

;

${this.gpuOutDataName}(cdi + ${this.gpuOutDataStartIndex}) := act;
`
			);
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			if (this.useBias) {
				Ment.webMonkeys.work(
					this.outSize(),
					`
float act = ${this.gpuBiasGradsName}(i) + ${this.gpuErrorArrayName}(i);

;

${this.gpuBiasGradsName}(i) := act;
`
				);
			}

			Ment.webMonkeys.work(
				this.filters * this.inDepth * this.filterWidth * this.filterHeight,
				`
int remaining = i;
int ifromi = int(remaining / (${this.inDepth * this.filterHeight * this.filterWidth}));
remaining = remaining - (ifromi * ${this.inDepth * this.filterHeight * this.filterWidth});
int hfromi = int(remaining / (${this.filterHeight * this.filterWidth}));
remaining = remaining - (hfromi * ${this.filterHeight * this.filterWidth});
int jfromi = int(remaining / (${this.filterWidth}));
remaining = remaining - (jfromi * ${this.filterWidth});
int kfromi = remaining;


int cdi = i;

int iFWIHID = ifromi * ${this.fWIHID};
int hFWIH = hfromi * ${this.fWIH} + iFWIHID;
int jFWHFWIH = jfromi * ${this.filterWidth} + hFWIH;
float act = ${this.gpuFilterWGrads}(kfromi + jFWHFWIH);
for (int vfromi = 0; vfromi < ${this.inHeight}; vfromi++) {
for (int cfromi = 0; cfromi < ${this.inWidth}; cfromi++) {
int acc = (((vfromi * ${this.inWidth} + cfromi)) * ${this.filterHeight * this.filterWidth});
if(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi) > 0.5){

int amnt = int(${this.gpuAccessedMap}(acc + jfromi * ${this.filterWidth} + kfromi)) - 1;
int gfromi = int(amnt / ${this.wMFWPO});
int bfromi = amnt - gfromi * ${this.wMFWPO};
int iHMFWMF = ifromi * ${this.hMFWMF};

int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};
int hWIH = hfromi * ${this.wIH};
int jGAIWBA = (jfromi + ga) * ${this.inWidth} + hWIH + ba;

act += ${this.gpuInDataName}(odi + ${this.gpuInDataStartIndex}) * ${this.gpuErrorArrayName}(kfromi + jGAIWBA);





}
}
};

;

${this.gpuFilterWGrads}(kfromi + jFWHFWIH) := act;
`
			);

			Ment.webMonkeys.work(
				this.filters * this.hMFHPO * this.wMFWPO,
				`
float act = 0.0;
int ifromi = int(i / ${this.hMFHPO * this.wMFWPO});
int gfromi = int((i - ifromi * ${this.hMFHPO * this.wMFWPO}) / ${this.wMFWPO});
int bfromi = int((i - ifromi * ${this.hMFHPO * this.wMFWPO} - gfromi * ${this.wMFWPO}));


int iHMFWMF = ifromi * ${this.hMFWMF};
int iFWIHID = ifromi * ${this.fWIHID};
int ga = gfromi * ${this.stride};
int gWMFWPO = gfromi * ${this.wMFWPO};
int odi = bfromi + gWMFWPO + iHMFWMF;
int ba = bfromi * ${this.stride};

for (int h = 0; h < ${this.inDepth}; h++) {
	int hWIH = h * ${this.wIH} + ba;
	int hFWIH = h * ${this.fWIH} + iFWIHID;

	for (int j = 0; j < ${this.filterHeight}; j++) {
		int jGAIWBA = (j + ga) * ${this.inWidth} + hWIH;
		int jFWHFWIH = j * ${this.filterWidth} + hFWIH;
		for (int k = 0; k < ${this.filterWidth}; k++) {
			act += ${this.gpuErrorArrayName}(k + jGAIWBA) * ${this.gpuFilterW}(k + jFWHFWIH);
		}
	}
}

;

${this.gpuCostsArrayName}(odi) := act;
				`
			);
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		getParamsAndGrads(forUpdate = true) {
			// if (forUpdate) {
			// 	if (ConvLayer.averageOutGrads) {
			// 		for (var i = 0; i < this.filterws.length; i++) {
			// 			this.filterws[i] /= this.filters;
			// 		}
			// 	}
			// }
			if (this.useBias) {
				return [this.gpuFilterW, this.gpuFilterWGrads, this.gpuBiasName, this.gpuBiasGradsName];
			} else {
				return [this.gpuFilterW, this.gpuFilterWGrads];
			}
		}

		save() {
			this.filterw = Ment.webMonkeys.get(this.gpuFilterW);
			this.b = Ment.webMonkeys.get(this.gpuBiasName);
			let ret = JSON.stringify(this, function (key, value) {
				if (
					key == "filterws" ||
					key == "gpuFilterWGrads" ||
					key == "gpuBiasGradsName" ||
					key == "gpuAccessedMap" ||
					key == "filterbs" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "accessed" ||
					key == "bs" ||
					key == "ws" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "hMFHPO" ||
					key == "wMFWPO" ||
					key == "hMFWMF" ||
					key == "wIH" ||
					key == "wIHID" ||
					key == "fWIH" ||
					key == "fWIHID" ||
					key == (this.useBias ? null : "b")
				) {
					return undefined;
				}

				return value;
			});
			delete this.filterw;
			delete this.b;

			return ret;
		}

		static load(json) {
			//inWidth, inHeight, inDepth, filterWidth, filterHeight, filters = 3, stride = 1,
			let saveObject = JSON.parse(json);
			let layer = new DeconvLayerGPU(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				[saveObject.filterWidth, saveObject.filterHeight],
				saveObject.filters,
				saveObject.stride,
				saveObject.useBias
			);
			let temp = new Float32Array(layer.filters * layer.inDepth * layer.filterWidth * layer.filterHeight);
			for (var i = 0; i < layer.filters * layer.inDepth * layer.filterWidth * layer.filterHeight; i++) {
				temp[i] = saveObject.filterw[i];
			}
			Ment.webMonkeys.set(layer.gpuFilterW, temp);
			if (layer.useBias) {
				temp = new Float32Array(layer.outSize());
				for (var i = 0; i < layer.outSize(); i++) {
					temp[i] = saveObject.b[i];
				}
				Ment.webMonkeys.set(layer.gpuBiasName, temp);
			}
			return layer;
		}
	}

	Ment.DeconvLayerGPU = DeconvLayerGPU;
	Ment.DeConvLayerGPU = DeconvLayerGPU;
	Ment.DeconvGPU = DeconvLayerGPU;
	Ment.DeConvGPU = DeconvLayerGPU;

	//only uncomment if you know what your doing
}

{
	class DepaddingLayerGPU {
		constructor(outDim, pad) {
			if (outDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let outWidth = outDim[0];
			let outHeight = outDim[1];
			let outDepth = outDim[2];

			pad = pad || 2;
			this.pad = pad;
			this.outWidth = outWidth;
			this.outHeight = outHeight;
			this.outDepth = outDepth;
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
		}

		forward() {
			// Ment.webMonkeys.fill(this.gpuOutDataName, this.padwith);
			Ment.webMonkeys.work(
				this.outSize(),
				// prettier-ignore
				`
float act = 0.0;
int remaining = i;
int ifromi = int(remaining / (${this.outHeight * this.outWidth}));
remaining = remaining - (ifromi * ${this.outHeight * this.outWidth});
int jfromi = int(remaining / (${this.outWidth}));
remaining = remaining - (jfromi * ${this.outWidth});
int hfromi = remaining;
int prop = ifromi * ${this.outHeight} * ${this.outWidth} + jfromi * ${this.outWidth} + hfromi;
int index = (jfromi + 1) * ${this.pad} * 2 + -${this.pad} + ${this.pad} * (${this.outWidth} + ${this.pad} * 2) + prop + ifromi * ((${this.outWidth} + ${this.pad} * 2) * ${this.pad}) * 2 + ifromi * (${this.outHeight} * ${this.pad}) * 2;

act = ${this.gpuInDataName}(index + ${this.gpuInDataStartIndex});

${this.gpuOutDataName}(prop + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}
		inSize() {
			return (this.outWidth + this.pad * 2) * (this.outHeight + this.pad * 2) * this.outDepth;
		}

		outSize() {
			return this.outWidth * this.outHeight * this.outDepth;
		}

		outSizeDimensions() {
			return [this.outWidth, this.outHeight, this.outDepth];
		}

		inSizeDimensions() {
			return [this.outWidth + this.pad * 2, this.outHeight + this.pad * 2, this.outDepth];
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}
			Ment.webMonkeys.work(
				this.outSize(),
				// prettier-ignore
				`
float act = 0.0;
int remaining = i;
int ifromi = int(remaining / (${this.outHeight * this.outWidth}));
remaining = remaining - (ifromi * ${this.outHeight * this.outWidth});
int jfromi = int(remaining / (${this.outWidth}));
remaining = remaining - (jfromi * ${this.outWidth});
int hfromi = remaining;
int prop = ifromi * ${this.outHeight} * ${this.outWidth} + jfromi * ${this.outWidth} + hfromi;
int index = (jfromi + 1) * ${this.pad} * 2 + -${this.pad} + ${this.pad} * (${this.outWidth} + ${this.pad} * 2) + prop + ifromi * ((${this.outWidth} + ${this.pad} * 2) * ${this.pad}) * 2 + ifromi * (${this.outHeight} * ${this.pad}) * 2;

act = ${this.gpuErrorArrayName}(prop);

${this.gpuCostsArrayName}(index) := act;
				`
			);
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "accessed"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new DepaddingLayerGPU([saveObject.outWidth, saveObject.outHeight, saveObject.outDepth], saveObject.pad);
			return layer;
		}
	}

	Ment.DePaddingLayerGPU = DepaddingLayerGPU;
	Ment.DepaddingLayerGPU = DepaddingLayerGPU;
	Ment.DepadLayerGPU = DepaddingLayerGPU;
	Ment.DepaddingGPU = DepaddingLayerGPU;
	Ment.DepadGPU = DepaddingLayerGPU;
}

{
	class FCLayerGPU {
		constructor(inSize, outSize, useBias) {
			this.useBias = useBias == undefined ? true : useBias;
			this.gpuWeightsName = Ment.makeid(8); //generates random 8 character string
			this.gpuWeightsGradsName = Ment.makeid(8); //generates random 8 character string
			this.gpuBiasName = Ment.makeid(8); //generates random 8 character string
			this.gpuBiasGradsName = Ment.makeid(8); //generates random 8 character string
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
			this.inSize = () => {
				return inSize;
			};

			this.outSize = () => {
				return outSize;
			};

			let temp = new Float32Array(inSize * outSize);
			for (var i = 0; i < inSize * outSize; i++) {
				temp[i] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1); //set random weights
			}
			Ment.webMonkeys.set(this.gpuWeightsName, temp);
			temp.fill(0);
			Ment.webMonkeys.set(this.gpuWeightsGradsName, temp);

			temp = new Float32Array(outSize);
			if (this.useBias) {
				for (var i = 0; i < outSize; i++) {
					temp[i] = 0.1 * Math.random() * (Math.random() > 0.5 ? -1 : 1); // set random bias
				}
			}
			Ment.webMonkeys.set(this.gpuBiasName, temp);
			temp.fill(0);
			Ment.webMonkeys.set(this.gpuBiasGradsName, temp);
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}
			Ment.webMonkeys.work(
				this.inSize(),
				`
float act = 0.0;
for (int j = 0; j < ${this.outSize()}; j++) {
	act += ${this.gpuWeightsName}(j + i * ${this.outSize()}) * ${this.gpuErrorArrayName}(j) * 2.0;
}
act = act / ${this.outSize()}.0;

;

${this.gpuCostsArrayName}(i) := act;
				`
			);

			//this code is so weird but works :\
			Ment.webMonkeys.work(
				this.inSize() * this.outSize(),
				`
float act = 0.0;
int j = i - (int(i / ${this.outSize()}) * ${this.outSize()});
int k = (i - j) / ${this.outSize()};
act = ${this.gpuInDataName}(k + ${this.gpuInDataStartIndex}) * ${this.gpuErrorArrayName}(j) * 2.0;
act += ${this.gpuWeightsGradsName}(i);

;

${this.gpuWeightsGradsName}(i) := act;
`
			);

			if (this.useBias) {
				Ment.webMonkeys.work(
					this.outSize(),
					`
float act = ${this.gpuBiasGradsName}(i) + ${this.gpuErrorArrayName}(i);

;

${this.gpuBiasGradsName}(i) := act;
`
				);
			}
		}

		getParamsAndGrads(forUpdate = true) {
			if (this.useBias) {
				return [this.gpuWeightsName, this.gpuWeightsGradsName, this.gpuBiasName, this.gpuBiasGradsName];
			} else {
				return [this.gpuWeightsName, this.gpuWeightsGradsName];
			}
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			this.savedInSize = this.inSize();
			this.savedOutSize = this.outSize();
			this.w = Ment.webMonkeys.get(this.gpuWeightsName);
			this.b = Ment.webMonkeys.get(this.gpuBiasName);
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "outData" ||
					key == "inData" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "gpuBiasName" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "gpuBiasGradsName" ||
					key == "gpuWeightsName" ||
					key == "gpuWeightsGradsName" ||
					key == (this.useBias ? null : "b")
				) {
					return undefined;
				}

				return value;
			});

			delete this.savedInSize;
			delete this.w;
			delete this.b;
			delete this.savedOutSize;

			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new FCLayerGPU(saveObject.savedInSize, saveObject.savedOutSize, saveObject.useBias);

			let temp = new Float32Array(layer.inSize() * layer.outSize());
			for (var i = 0; i < layer.inSize() * layer.outSize(); i++) {
				temp[i] = saveObject.w[i];
			}
			Ment.webMonkeys.set(layer.gpuWeightsName, temp);
			if (layer.useBias) {
				temp = new Float32Array(layer.outSize());
				for (var i = 0; i < layer.outSize(); i++) {
					temp[i] = saveObject.b[i];
				}
				Ment.webMonkeys.set(layer.gpuBiasName, temp);
			}
			return layer;
		}

		// initGPUActivations(
		// 	inDataGpuArrayName,
		// 	outDataGpuArrayName,
		// 	costsGpuArrayName,
		// 	errorGpuArrayName,
		// 	inDataGPUStartIndex,
		// 	outDataGPUStartIndex
		// ) {
		// 	//i know its confusing but costs are the erorr of the indata neurons and error is the error of the outdata neurons
		// 	if (!inDataGpuArrayName) {
		// 		this.gpuInDataName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!outDataGpuArrayName) {
		// 		this.gpuOutDataName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!costsGpuArrayName) {
		// 		this.gpuCostsArrayName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!errorGpuArrayName) {
		// 		this.gpuErrorArrayName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	this.gpuInDataName = inDataGpuArrayName;
		// 	this.gpuOutDataName = outDataGpuArrayName;
		// 	this.gpuCostsArrayName = costsGpuArrayName;
		// 	this.gpuErrorArrayName = errorGpuArrayName;
		// 	this.gpuInDataStartIndex = inDataGPUStartIndex;
		// 	this.gpuOutDataStartIndex = outDataGPUStartIndex;

		// 	let temp = new Float32Array(this.inSize());
		// 	temp.fill(0);
		// 	Ment.webMonkeys.set(this.gpuInDataName, temp);
		// 	Ment.webMonkeys.set(this.gpuCostsArrayName, temp);

		// 	temp = new Float32Array(this.outSize());
		// 	temp.fill(0);
		// 	Ment.webMonkeys.set(this.gpuOutDataName, temp);
		// 	Ment.webMonkeys.set(this.gpuErrorArrayName, temp);
		// }

		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = 0.0;
for (int j = 0; j < ${this.inSize()}; j++) {
	act += ${this.gpuInDataName}(j + ${this.gpuInDataStartIndex}) * ${this.gpuWeightsName}(i + j * ${this.outSize()});
}
act += ${this.gpuBiasName}(i);
${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
							`
			);
		}
	}

	Ment.FCLayerGPU = FCLayerGPU;
	Ment.FCGPU = FCLayerGPU;
}

{
	class IdentityLayerGPU {
		//this layer outputs the inputs with no changes
		constructor(size) {
			this.nextLayer; //the connected layer
			this.pl; //reference to previous layer
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
			this.inSize = () => {
				return size;
			};

			this.outSize = () => {
				return size;
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

		get previousLayer() {
			return this.pl;
		}
		set previousLayer(layer) {
			// try to connect to it
			this.inSize = () => {
				return layer.outSize();
			};

			this.outSize = () => {
				return layer.outSize();
			};
			this.pl = layer;
		}
		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
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

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "pl" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
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
			let layer = new IdentityLayerGPU(saveObject.savedSize);
			return layer;
		}
	}

	Ment.IdentityLayerGPU = IdentityLayerGPU;
	Ment.IdentityGPU = IdentityLayerGPU;
	Ment.InputGPU = IdentityLayerGPU;
	Ment.DummyLayerGPU = IdentityLayerGPU;
	Ment.DummyGPU = IdentityLayerGPU;
	Ment.InputLayerGPU = IdentityLayerGPU;
	Ment.OutputLayerGPU = IdentityLayerGPU;
	Ment.OutputGPU = IdentityLayerGPU;
}

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
			// 	this.costs = new Float32Array(layer.outSize());
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

${this.gpuCostsArrayName}(i) := act;
				`
			);
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
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
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
			let layer = new InputInsertionLayerGPU(saveObject.savedSize, saveObject.savedInputSize);
			return layer;
		}
	}

	Ment.InputInsertionLayerGPU = InputInsertionLayerGPU;
	Ment.InputInsertionGPU = InputInsertionLayerGPU;
	Ment.ActivationInsertionGPU = InputInsertionLayerGPU;
}

{
	class LeakyReluLayerGPU extends Ment.ActivationBaseGPU {
		static leakySlope = 0.01;

		constructor(size) {
			super(size);
		}

		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = ${this.gpuInDataName}(${this.gpuInDataStartIndex} + i);

if(act < 0.0){
	act = act * ${LeakyReluLayerGPU.leakySlope};
}

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = ${this.gpuErrorArrayName}(i);

if(${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) < 0.0){
	act = act * ${LeakyReluLayerGPU.leakySlope};
}

;

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new LeakyReluLayerGPU(saveObject.savedSize);
			return layer;
		}
	}
	Ment.LeakyReluLayerGPU = LeakyReluLayerGPU;
	Ment.LeakyReluGPU = LeakyReluLayerGPU;
	Ment.LReluGPU = LeakyReluLayerGPU;
}

{
	class PaddingLayerGPU {
		constructor(inDim, pad, padwith) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			pad = pad || 2;
			padwith = padwith || 0;
			this.pad = pad;
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.padwith = padwith;
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
		}

		forward() {
			Ment.webMonkeys.work(
				this.inSize(),
				// prettier-ignore
				`
float act = 0.0;
int remaining = i;
int ifromi = int(remaining / (${this.inHeight * this.inWidth}));
remaining = remaining - (ifromi * ${this.inHeight * this.inWidth});
int jfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (jfromi * ${this.inWidth});
int hfromi = remaining;
int prop = ifromi * ${this.inHeight} * ${this.inWidth} + jfromi * ${this.inWidth} + hfromi;
int index = (jfromi + 1) * ${this.pad} * 2 + -${this.pad} + ${this.pad} * (${this.inWidth} + ${this.pad} * 2) + prop + ifromi * ((${this.inWidth} + ${this.pad} * 2) * ${this.pad}) * 2 + ifromi * (${this.inHeight} * ${this.pad}) * 2;

act = ${this.gpuInDataName}(prop + ${this.gpuInDataStartIndex});

${this.gpuOutDataName}(index + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}
			Ment.webMonkeys.work(
				this.inSize(),
				// prettier-ignore
				`
float act = 0.0;
int remaining = i;
int ifromi = int(remaining / (${this.inHeight * this.inWidth}));
remaining = remaining - (ifromi * ${this.inHeight * this.inWidth});
int jfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (jfromi * ${this.inWidth});
int hfromi = remaining;
int prop = ifromi * ${this.inHeight} * ${this.inWidth} + jfromi * ${this.inWidth} + hfromi;
int index = (jfromi + 1) * ${this.pad} * 2 + -${this.pad} + ${this.pad} * (${this.inWidth} + ${this.pad} * 2) + prop + ifromi * ((${this.inWidth} + ${this.pad} * 2) * ${this.pad}) * 2 + ifromi * (${this.inHeight} * ${this.pad}) * 2;

act = ${this.gpuErrorArrayName}(index);

${this.gpuCostsArrayName}(prop) := act;
				`
			);
		}

		inSize() {
			return this.inWidth * this.inHeight * this.inDepth;
		}

		outSize() {
			return (this.inWidth + this.pad * 2) * (this.inHeight + this.pad * 2) * this.inDepth;
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [this.inWidth + this.pad * 2, this.inHeight + this.pad * 2, this.inDepth];
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new PaddingLayerGPU(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				saveObject.pad,
				saveObject.padwith
			);
			return layer;
		}
	}

	Ment.PaddingLayerGPU = PaddingLayerGPU;
	Ment.PadLayerGPU = PaddingLayerGPU;
	Ment.PaddingGPU = PaddingLayerGPU;
	Ment.PadGPU = PaddingLayerGPU;
}

{
	class ResEmitterLayerGPU {
		//this layer outputs the inputs with no changes
		constructor(id) {
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.receiver; // a reference to the receiver layer so we can skip layers
			//this will be set by the receiver  when the net is initialized
			this.gpuCostsFromEmitterName; //gets set by the receiver
			this.pl = undefined;
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inSize = () => {
				return layer.outSize();
			};

			this.outSize = () => {
				return layer.outSize();
			};
			this.pl = layer;
		}

		// initGPUActivations(inDataGpuArrayName, outDataGpuArrayName, costsGpuArrayName, errorGpuArrayName) {
		// 	//i know its confusing but costs are the erorr of the indata neurons and error is the error of the outdata neurons
		// 	if (!inDataGpuArrayName) {
		// 		this.gpuInDataName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!outDataGpuArrayName) {
		// 		this.gpuOutDataName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!costsGpuArrayName) {
		// 		this.gpuCostsArrayName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	if (!errorGpuArrayName) {
		// 		this.gpuErrorArrayName = Ment.makeid(8); //generates random 8 character string
		// 	}
		// 	this.gpuInDataName = inDataGpuArrayName;
		// 	this.gpuOutDataName = outDataGpuArrayName;
		// 	this.gpuCostsArrayName = costsGpuArrayName;
		// 	this.gpuErrorArrayName = errorGpuArrayName;

		// 	let temp = new Float32Array(this.inSize());
		// 	temp.fill(0);
		// 	Ment.webMonkeys.set(this.gpuInDataName, temp);
		// 	Ment.webMonkeys.set(this.gpuCostsArrayName, temp);

		// 	temp = new Float32Array(this.outSize());
		// 	temp.fill(0);
		// 	Ment.webMonkeys.set(this.gpuOutDataName, temp);
		// 	Ment.webMonkeys.set(this.gpuErrorArrayName, temp);

		// 	this.receiver.gpuInDataFromEmitterName = this.gpuOutDataName;
		// }

		forward() {
			//the outData of this layer is the same object referenced in the inData of the Receiver layer

			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				//this code should only run if user is dumb, but just in case
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			Ment.webMonkeys.work(
				this.inSize(),
				`
float act = (${this.gpuErrorArrayName}(i) + ${this.gpuCostsFromEmitterName}(i)) / 2.0;

;

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}
		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "receiver" ||
					key == "pl" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "gpuCostsFromEmitter" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "emitter"
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
			let layer = new ResEmitterLayerGPU(saveObject.id);
			return layer;
		}
	}

	Ment.ResEmitterLayerGPU = ResEmitterLayerGPU;
	Ment.ResEmitterGPU = ResEmitterLayerGPU;
	Ment.ResEGPU = ResEmitterLayerGPU;
}

{
	class ResReceiverLayerGPU {
		//this layer outputs the inputs with no changes
		constructor(id, mode = "concat") {
			//mode can be concat or add or average
			this.mode = mode;
			this.id = id || 0;
			this.nextLayer; //the connected layer
			this.emitter;
			this.gpuCostsForEmitter;
			this.pl; // holds a reference to previous layer
		}

		get previousLayer() {
			return this.pl;
		}

		set previousLayer(layer) {
			this.inSize = () => {
				return layer.outSize();
			};

			this.pl = layer;

			//time to find this layers soulmate
			let found = false;
			let currentLayer = layer; //start at this layer go backward until find a emitter with the same ID
			while (!found) {
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.ResEmitterGPU) {
					found = true;
				} else {
					currentLayer = currentLayer.previousLayer;
					if (currentLayer == undefined) {
						throw "Could not find Matching Emitter Layer for Receiver Layer ID: " + this.id;
					}
				}
			}
			this.emitter = currentLayer; //so they can find each other again :)
			currentLayer.receiver = this;
			if (this.mode == "add" || this.mode == "average") {
				if (layer.outSize() != this.emitter.outSize()) {
					throw "emitter size must equal the size of the previous layer of the corresponding receiver layer";
				}
				this.outSize = () => {
					return layer.outSize();
				};
			} else if (this.mode == "concat") {
				this.outSize = () => {
					return layer.outSize() + this.emitter.outSize();
				};
			}
			this.gpuCostsForEmitterName = Ment.makeid(8);
			Ment.webMonkeys.set(this.gpuCostsForEmitterName, this.emitter.outSize());
			this.emitter.gpuCostsFromEmitterName = this.gpuCostsForEmitterName;
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		//we dont look in front because residual data only goes forwards.

		forward() {
			//the outData of this layer is the same object referenced in the inData of the Receiver layer

			if (this.mode == "concat") {
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
				);
				Ment.webMonkeys.work(
					this.emitter.outSize(),
					`
float act = ${this.emitter.gpuOutDataName}(i + ${this.emitter.gpuOutDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.inSize()} + ${this.gpuOutDataStartIndex}) := act;
				`
				);
			} else if (this.mode == "add") {
				Ment.webMonkeys.work(
					this.outSize(),
					`
float act = ${this.emitter.gpuOutDataName}(i + ${this.emitter.gpuOutDataStartIndex}) + ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex});

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
				);
			} else if (this.mode == "average") {
				Ment.webMonkeys.work(
					this.outSize(),
					`
float act = (${this.emitter.gpuOutDataName}(i + ${this.emitter.gpuOutDataStartIndex}) + ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex})) / 2.0;

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
				);
			}
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}

			if (this.mode == "concat") {
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i);

;

${this.gpuCostsArrayName}(i) := act;
				`
				);
				Ment.webMonkeys.work(
					this.emitter.outSize(),
					`
float act = ${this.gpuErrorArrayName}(i + ${this.inSize()});

;

${this.gpuCostsForEmitterName}(i) := act;				`
				);
			} else if (this.mode == "add") {
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i) / 2.0;

;

${this.gpuCostsArrayName}(i) := act;
				`
				);
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i) / 2.0;

;

${this.gpuCostsForEmitterName}(i) := act;				`
				);
			} else if (this.mode == "average") {
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i);

;

${this.gpuCostsArrayName}(i) := act;
				`
				);
				Ment.webMonkeys.work(
					this.inSize(),
					`
float act = ${this.gpuErrorArrayName}(i);

;

${this.gpuCostsForEmitterName}(i) := act;				`
				);
			}
		}

		save() {
			this.savedSize = this.inSize();

			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "receiver" ||
					key == "pl" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "gpuCostsForEmitter" ||
					key == "gpuInDataFromEmitterName" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "emitter"
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
			let layer = new ResReceiverLayerGPU(saveObject.id, saveObject.mode);
			return layer;
		}
	}

	Ment.ResReceiverLayerGPU = ResReceiverLayerGPU;
	Ment.ResReceiverGPU = ResReceiverLayerGPU;
	Ment.ResRGPU = ResReceiverLayerGPU;
}

{
	class SigmoidLayerGPU extends Ment.ActivationBaseGPU {
		constructor(size) {
			super(size);
		}

		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = 1.0 / (1.0 + exp(-${this.gpuInDataName}(i + ${this.gpuInDataStartIndex})));

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
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
float zee = exp(-${this.gpuInDataName}(i + ${this.gpuInDataStartIndex}));
float act = ${this.gpuErrorArrayName}(i) * (zee / ((1.0 + zee) * (1.0 + zee)));

;

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new SigmoidLayerGPU(saveObject.savedSize);
			return layer;
		}
	}

	Ment.SigmoidLayerGPU = SigmoidLayerGPU;
	Ment.SigmoidGPU = SigmoidLayerGPU;
	Ment.SigGPU = SigmoidLayerGPU;
}

{
	class TanhLayerGPU extends Ment.ActivationBaseGPU {
		tanh(z) {
			return Math.tanh(z);
		}

		tanhPrime(z) {
			return 1 - Math.pow(Math.tanh(z), 2);
		}

		constructor(size) {
			super(size);
		}

		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float e = 2.71828;
float act = 2.0 / (1.0 + pow(e,-2.0 * ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex}))) - 1.0;

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
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
float e = 2.71828;
float zee = 2.0 / (1.0 + pow(e,-2.0 * ${this.gpuInDataName}(i + ${this.gpuInDataStartIndex}))) - 1.0;
float act = 1.0 - (zee * zee);
act = act * ${this.gpuErrorArrayName}(i);
;

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new TanhGPU(saveObject.savedSize);
			return layer;
		}
	}

	Ment.TanhLayerGPU = TanhLayerGPU;
	Ment.TanhGPU = TanhLayerGPU;
}

{
	class UpscalingLayerGPU {
		constructor(inDim, scale) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];

			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.outWidth = inWidth * scale;
			this.outHeight = inHeight * scale;
			this.scale = scale;
			this.nextLayer; //the connected layer
			this.previousLayer; //the previousLayer
		}
		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [this.outWidth, this.outHeight, this.inDepth];
		}
		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = 0.0;

int remaining = i;
int ifromi = int(remaining / (${this.inHeight * this.scale * this.inWidth * this.scale}));
remaining = remaining - (ifromi * ${this.scale * this.inHeight * this.inWidth * this.scale});
int hfromi = int(remaining / (${this.inWidth * this.scale}));
remaining = remaining - (hfromi * ${this.inWidth * this.scale});
int jfromi = remaining;

act += ${this.gpuInDataName}(${this.gpuInDataStartIndex} + ifromi * ${this.inHeight * this.inWidth} + int(hfromi / ${
					this.scale
				}) * ${this.inWidth} + int(jfromi / ${this.scale}))

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
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
float act = 0.0;

int remaining = i;
int ifromi = int(remaining / (${this.inHeight * this.inWidth}));
remaining = remaining - (ifromi * ${this.inHeight * this.inWidth});
int hfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (hfromi * ${this.inWidth});
int jfromi = remaining;

for(int c = 0; c < ${this.scale}; c++){
for(int b = 0; b < ${this.scale}; b++){

act += ${this.gpuErrorArrayName}(ifromi * ${this.outHeight * this.outWidth} + (((hfromi * ${this.scale}) + c) * ${
					this.outWidth
				}) + (jfromi * ${this.scale} + b));

};
};

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		inSize() {
			return this.inHeight * this.inWidth * this.inDepth;
		}

		outSize() {
			return this.outHeight * this.outWidth * this.inDepth;
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "ws" ||
					key == "bs" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
				) {
					return undefined;
				}

				return value;
			});

			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new UpscalingLayerGPU([saveObject.inWidth, saveObject.inHeight, saveObject.inDepth], saveObject.scale);
			return layer;
		}
	}

	Ment.UpscalingLayerGPU = UpscalingLayerGPU;
	Ment.UpscalingGPU = UpscalingLayerGPU;
	Ment.UpscaleGPU = UpscalingLayerGPU;
	Ment.UpScalingGPU = UpscalingLayerGPU;
	Ment.UpscalingLayerGPU = UpscalingLayerGPU;
}

{
	/*
        Gives the gradients "momentum" down the slope. Nothing special to it but
		it is rumored that it increased training speed a lot! I think it might be 
		a good idea to use this instead of SGD

		Super excited about it. 
    */
	class Momentum {
		constructor(momentum = 0.9) {
			this.netObject; //will be set to the net object
			this.lastGrads = [];
			this.momentum = momentum; // a constant which represents how strong the gradients "momentum" will be
		} //END OF CONSTRUCTOR

		initialize() {
			//we need to get "this.lastGrads" list the right size. It needs to be as long as there are gradients in this network.
			let pag = this.netObject.getParamsAndGrads();
			var gradCounter = 0;
			for (var j = 0; j < pag.length; j += 2) {
				let grads = pag[j + 1];
				gradCounter += grads.length;
			}
			this.lastGrads = new Float32Array(gradCounter);
			this.lastGrads.fill(0);
		}
		applyGradients(paramsAndGrads) {
			var gradCounter = 0;
			for (var j = 0; j < paramsAndGrads.length; j += 2) {
				let params = paramsAndGrads[j];
				let grads = paramsAndGrads[j + 1];
				for (var k = 0; k < params.length; k++) {
					params[k] +=
						(grads[k] / this.netObject.iteration) * this.netObject.learningRate + this.momentum * this.lastGrads[gradCounter];
					this.lastGrads[gradCounter] =
						(grads[k] / this.netObject.iteration) * this.netObject.learningRate + this.momentum * this.lastGrads[gradCounter];
					grads[k] = 0;
					gradCounter++;
				}
			}
		}
	} //END OF Momentum CLASS DECLARATION

	Ment.Momentum = Momentum;
}

{
	/*
        Gives the gradients "momentum" down the slope. Nothing special to it but
		it is rumored that it increased training speed a lot! I think it might be 
		a good idea to use this instead of SGD

		Super excited about it. 
    */
	class MomentumGPU {
		constructor(momentum = 0.9) {
			this.netObject; //will be set to the net object
			this.lastGrads = [];
			this.momentum = momentum; // a constant which represents how strong the gradients "momentum" will be
		} //END OF CONSTRUCTOR

		initialize() {
			//we need to get "this.lastGrads" list the right size. It needs to be as long as there are gradients in this network.
			let pag = this.netObject.getParamsAndGrads();
			for (var j = 0; j < pag.length; j += 2) {
				let grads = pag[j + 1];
				Ment.webMonkeys.set(this.lastGrads[this.lastGrads.push(Ment.makeid(8)) - 1], Ment.webMonkeys.getLen(grads));
			}
		}
		applyGradients(paramsAndGrads) {
			for (var j = 0; j < paramsAndGrads.length; j += 2) {
				let params = paramsAndGrads[j];
				let grads = paramsAndGrads[j + 1];
				// for (var k = 0; k < params.length; k++) {
				// 	params[k] += (grads[k] / this.netObject.iteration) * this.netObject.learningRate + this.momentum * this.lastGrads[j][k];
				// 	this.lastGrads[gradCounter] =
				// 		(grads[k] / this.netObject.iteration) * this.netObject.learningRate + this.momentum * this.lastGrads[j][k];
				// 	grads[k] = 0;
				// 	gradCounter++;
				// }
				Ment.webMonkeys.work(
					Ment.webMonkeys.getLen(params),
					`
float act = ${params}(i) + (${grads}(i) / ${parseFloat(this.netObject.batchSize).toFixed(1)}) * ${parseFloat(
						this.netObject.learningRate
					).toFixed(8)} + ${parseFloat(this.momentum).toFixed(8)} * ${this.lastGrads[j / 2]}(i);

;

${params}(i) := act;

`
				);

				Ment.webMonkeys.work(
					Ment.webMonkeys.getLen(params),
					`
float act = (${grads}(i) / ${parseFloat(this.netObject.batchSize).toFixed(8)}) * ${parseFloat(
						this.netObject.learningRate
					).toFixed(8)} + ${parseFloat(this.momentum).toFixed(8)} * ${this.lastGrads[j / 2]}(i);

;

${this.lastGrads[j / 2]}(i) := act;

`
				);

				Ment.webMonkeys.fill(grads, 0);
			}
		}
	} //END OF Momentum CLASS DECLARATION

	Ment.MomentumGPU = MomentumGPU;
}

{
	/*
        This is the basic white girl of optimizers. Doesn't even do anything to the gradient.
        batch size is already built in to the network so we dont even have to do that here
    */
	class SGD {
		constructor() {
			this.netObject; //will be set to the net object
		} //END OF CONSTRUCTOR

		//this function gets called after this.netObject is set by the network.
		initialize() {}

		applyGradients(paramsAndGrads) {
			for (var j = 0; j < paramsAndGrads.length; j += 2) {
				let params = paramsAndGrads[j];
				let grads = paramsAndGrads[j + 1];
				for (var k = 0; k < params.length; k++) {
					params[k] += (grads[k] / this.netObject.iteration) * this.netObject.learningRate;
					grads[k] = 0;
				}
			}
		}
	} //END OF SGD CLASS DECLARATION

	Ment.SGD = SGD;
}

{
	/*
       SGD except it works for GPU AI instead of cpu ai
    */
	class SGDGPU {
		constructor() {
			this.netObject; //will be set to the net object
		} //END OF CONSTRUCTOR

		//this function gets called after this.netObject is set by the network.
		initialize() {}

		applyGradients(paramsAndGrads) {
			for (var j = 0; j < paramsAndGrads.length; j += 2) {
				let params = paramsAndGrads[j];
				let grads = paramsAndGrads[j + 1];
				// params[k] += (grads[k] / this.netObject.iteration) * this.netObject.learningRate;
				// grads[k] = 0;
				let len = Ment.webMonkeys.getLen(params);
				Ment.webMonkeys.work(
					len,
					`
float act = ${params}(i) + (${grads}(i) / ${parseFloat(this.netObject.batchSize).toFixed(8)}) * ${parseFloat(
						this.netObject.learningRate
					).toFixed(8)};

;

${params}(i) := act;

`
				);
				Ment.webMonkeys.fill(grads, 0);
			}
		}
	} //END OF SGD CLASS DECLARATION

	Ment.SGDGPU = SGDGPU;
}
