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
				layer.gpuOutDataName = state;
			}
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
				if (this.trainingMode) {
					Ment.webMonkeys.work(
						this.layers[0].inSize(),
						`
${this.layers[0].gpuInDataName}(i + ${this.layers[0].gpuInDataStartIndex}) := ${this.savedStates[this.currentState - 1]}(i + ${
							this.layers[this.layers.length - 1].gpuOutDataStartIndex
						});
					`
					);
				} else {
					Ment.webMonkeys.work(
						this.layers[0].inSize(),
						`
${this.layers[0].gpuInDataName}(i + ${this.layers[0].gpuInDataStartIndex}) := ${
							this.layers[this.layers.length - 1].gpuOutDataName
						}(i + ${this.layers[this.layers.length - 1].gpuOutDataStartIndex});
					`
					);
				}
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
			// 		layer.savedGradsForEmitter.fill(0);
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
			//backprop the error (first layers grads) from the "next iteration" (n+1).
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
				// 	expected[i] = this.layers[0].grads[i]; //from now on "expected" will represent backprop gradient or error.
				// }
				// backPropErrorInstead = true;
				Ment.webMonkeys.work(
					this.layers[0].inSize(),
					`
${this.layers[this.layers.length - 1].gpuErrorArrayName}(i) := ${this.layers[0].gpuGradsArrayName}(i);
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
