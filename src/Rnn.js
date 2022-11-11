var Ment = Ment || {};
{
	class Rnn extends Ment.Net{
		constructor(layers, optimizer) {
			console.log('DONT USE THIS YET RNN IS DEFINITLY NOT FINISHED AND WONT WORK.')
			super(layers,optimizer);
			this.savedStates = []; //fills with arrays of arrays containing in/out Datas for the layers. need to save them
			//to do backprop over time
		} //END OF CONSTRUCTOR

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

		setState(ind){
			if(ind == undefined){
				ind = this.savedStates.length - 1;
			}
			let state = this.savedStates[ind];

			for(var i = 0;i<this.layers.length;i++){
				let layer = this.layers[i];
				layer.inData = state[i];
				if(i > 0){
					this.layers[i-1].outData = layer.inData;
				}
			}
			this.layers[this.layers.length-1].outData = state[state.length - 1];

			//special cases
			for(var i = 0;i<this.layers.length;i++){
				if(this.layers[i].constructor.name == "ResReceiverLayer"){
					this.layers[i].inDataFromEmitter = this.layers[i].emitter.outData;
				}
			}

		}

		forward(data) {

			if(data == undefined){
				data = new Float32Array(this.layers[this.layers.length - 1].outData);
			}
			let ret = super.forward(data);
			//save the state
			let state = this.savedStates[this.savedStates.push([]) - 1];
			for(var i = 0;i<this.layers.length;i++){
				state.push(new Float32Array(this.layers[i].inData));
			}
			state.push(new Float32Array(this.layers[this.layers.length-1].outData));
			return ret;
		}

		resetRecurrentData() {
			for(var i = 0;i<this.layers.length;i++){
				let layer = this.layers[i];
				if(layer.constructor.name == "RecEmitterLayer"){
					layer.savedOutData.fill(0);
				}

				if(layer.constructor.name == "RecReceiverLayer"){
					layer.savedCostsForEmitter.fill(0);
				}
			}
		}



		train(input, expectedOut) {
			console.log('dont');
			this.forward(input);

			let loss = this.backward(expectedOut);
			//done backpropping
			return loss;
		}

		backward(expected) {
			//in rnns we might not have an expected array.. in this case.. 
			//backprop the error from the sequence.. yknow like how an rnn works.
			//this will only work if the indata and outdata of the network are the same.

			//you could have a network with different in/out sizes but you would need to specify
			//the expected input and output at each step. - Trevor 
			if(expected == undefined){
				if(this.layers[0].inSize() != this.layers[this.layers.length-1].outSize()){
					throw "uh oh, you made a FUCKY WUCKY. if your gonna train like that mister..." +
					"Then you better make sure the in size of your network is the same as the outsize!" +
					"or else daddy is gonna be extra mad! Now go ahead and search up how an RNN works buddy. Then we will talk";
				}
				expected = new Float32Array(this.layers[0].inSize());	
				for(var i = 0;i<this.layers[0].inSize();i++){
					expected[i] = this.layers[0].inData[i] + this.layers[0].costs[i];
				} //rip slow ass copy
			}
			this.setState();
			//custom code:
			this.layers[this.layers.length - 1].b = this.layers[0].inData;

			let ret = super.backward(expected);
			this.savedStates.pop();
			return ret;
		}

		static load(json) {
			let jsonObj = JSON.parse(json);
			let layers = [];
			for (var i = 0; i < jsonObj.layerAmount; i++) {
				let layer = Ment[jsonObj['layer' + i].type].load(jsonObj['layer' + i].layerData);
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
