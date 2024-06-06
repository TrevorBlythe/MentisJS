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
				state.push(new Float64Array(this.layers[i].inData));
			}
			state.push(new Float64Array(this.layers[this.layers.length - 1].outData));
		}

		forward(data) {
			if (data == undefined) {
				//todo you dont need to make a whole new array here it should get copied by the layers forward function
				data = new Float64Array(this.layers[this.layers.length - 1].outData);
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
					layer.savedGradsForEmitter.fill(0);
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
			//backprop the error (first layers grads) from the "next iteration" (n+1).
			//this will only work if the indata and outdata of the network are the same size.
			//This means you must specify a Final output at least, which makes sense.
			//you could have a network with different in/out sizes but you would need to specify
			//the expected input and output at each step. - Trevor
			if (expected == undefined) {
				if (this.layers[0].inSize() != this.layers[this.layers.length - 1].outSize()) {
					throw "Size Mismatch in RNN. In data and out data are different dimensions.";
				}
				expected = new Float64Array(this.layers[0].inSize()); //maybe could just set it as reference instead of copy???
				for (var i = 0; i < this.layers[0].inSize(); i++) {
					expected[i] = this.layers[0].grads[i]; //from now on "expected" will represent backprop gradient or error.
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
			let ret = new Ment.Rnn(layers, sus);
			ret.learningRate = jsonObj.learningRate;
			ret.batchSize = jsonObj.batchSize;
			return ret;
		} //end of load method
	} //END OF Rnn CLASS DECLARATION

	Ment.Rnn = Rnn;
}
