{
	class RecReceiverLayer {
		//Recurrent Receiver
		constructor(id, mode = 'concat') { //mode can be concat or add
			console.log('this layer isnt finished yet it wont work.... sory :(');
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
			this.nl; //guess
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
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.ResEmitter) {
					found = true;
				} else {
					currentLayer = currentLayer.previousLayer;
					if (currentLayer == undefined) {
						if(this.checkedFront){
							throw "COULD NOT FIND MATCHING REC EMITTER FOR REC RECEIVER LAYER";
						}
						this.checkedBehind = true;
						return; //give up for now
					}
				}
			}
			this.emitter = currentLayer;
			this.inDataFromEmitter = this.emitter.savedOutData;
			currentLayer.receiver = this; //so they can find each other again :)
			if(this.mode == 'add'){
				if(layer.outSize() != this.emitter.outSize()){
					throw "emitter size must equal the size of the previous layer of the corresponding receiver layer";
				}
				this.outData = new Float32Array(layer.outSize());
			}else if(this.mode == 'concat'){
				this.outData = new Float32Array(layer.outSize() + this.emitter.outSize());
			}
			this.costsForEmitter = new Float32Array(this.emitter.outSize());
		}

		get nextLayer() {
			return this.pl;
		}

		set nextLayer(layer) {
			this.nl = layer;
			//time to find this layers soulmate
			let found = false;
			let currentLayer = layer; //start at this layer go back until find a reciever with the same ID
			while (!found) {
				if (currentLayer.id == this.id && currentLayer.constructor == Ment.ResEmitter) {
					found = true;
				} else {
					currentLayer = currentLayer.nextLayer;
					if (currentLayer == undefined) {
						if(this.checkedBehind){
							throw "COULD NOT FIND MATCHING REC EMITTER FOR REC RECEIVER LAYER";
						}
						this.checkedFront = true;
						return; //give up for now
					}
				}
			}
			this.emitter = currentLayer;
			this.inDataFromEmitter = this.emitter.savedOutData;
			currentLayer.receiver = this; //so they can find each other again :)
			if(this.mode == 'add'){
				if(layer.outSize() != this.emitter.outSize()){
					throw "emitter size must equal the size of the previous layer of the corresponding receiver layer";
				}
				this.outData = new Float32Array(layer.outSize());
			}else if(this.mode == 'concat'){
				this.outData = new Float32Array(layer.outSize() + this.emitter.outSize());
			}
			this.costsForEmitter = new Float32Array(this.emitter.outSize());
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
			if(this.mode == 'concat'){
				for (var h = 0; h < this.inData.length; h++) {
					this.outData[h] = this.inData[h];
				}
				for (var h = this.inData.length; h < this.inData.length + this.inDataFromEmitter.length; h++) {
					this.outData[h] = this.inDataFromEmitter[h - this.inData.length];
				}
			}else if(this.mode == 'add'){
				for(var i = 0;i<this.outData.length;i++){
					this.outData[i] = this.inData[i] + this.inDataFromEmitter[i];
				}
			}
		}

		backward(expected) {
			let loss = 0;

			let getErr = (ind) => {
				return expected[i] - this.outData[i];
			}

			if(!expected){
				if (this.nextLayer == undefined) {
					throw 'nothing to backpropagate!';
				}
				getErr = (ind) => {
					return 	this.nextLayer.costs[ind]
				}
			}


			if(this.mode == 'concat'){
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = getErr(i);
					loss += this.costs[i];
				}
				for (var i = this.inData.length; i < this.inData.length + this.inDataFromEmitter.length; i++) {
					this.costsForEmitter[i - this.inData.length] = getErr(i);
					loss += this.costsForEmitter[i - this.inData.length];
				}
			} else if(this.mode == 'add'){
				for (var i = 0; i < this.inData.length; i++) {
					this.costs[i] = getErr(i);
					this.costsForEmitter[i] = getErr(i);
					loss += this.costs[i]; 
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
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (key == 'emitter' || key == 'pl' || key == 'receiver' || key == 'inData' || key == 'outData' || key == 'costs' || key == 'nextLayer' || key == 'previousLayer') {
					return undefined;
				}

				return value;
			});
			return ret;
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new RecReceiverLayer(saveObject.id,saveObject.mode);
			return layer;
		}
	}

	Ment.RecReceiverLayer = RecReceiverLayer;
	Ment.RecReceiver = RecReceiverLayer;
	Ment.RecR = RecReceiverLayer;

}
