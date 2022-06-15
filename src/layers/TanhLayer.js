
{

	class TanhLayer {

		tanh(z) {
			return Math.tanh(z);
		}
	
		tanhPrime(z) {
			return 1 - Math.pow(Math.tanh(z),2);
		}


		constructor(size) {
				this.nextLayer; //the connected layer
				this.inData = new Float32Array(size); 
				this.outData = new Float32Array(size); 
				this.costs = new Float32Array(size); //costs for each neuron
		}

		forward(inData){
					if(inData){
						if(inData.length != this.inData.length){
							throw "INPUT SIZE WRONG ON Tanh LAYER:\nexpected size ("
							 + this.inSize + "), got: (" + inData.length + ")";
						}
						for(var i = 0;i<inData.length;i++){
							this.inData[i] = inData[i];
						}					
					}

					for (var h = 0; h < this.outSize(); h++) {
						this.outData[h] = this.tanh(this.inData[h]);
					}
					//Oh the misery
		}

		backward(expected){
			if(!expected){
				if(this.nextLayer == undefined){
					throw "nothing to backpropagate!"
				}
				expected = [];
				for(var i = 0;i<this.outData.length;i++){
					expected.push(this.nextLayer.costs[i] + this.nextLayer.inData[i]);
				}
			}


			for(var j = 0;j<this.outSize();j++){
				let err = (expected[j] - this.outData[j]);
				// this.costs[j] = err * this.tanhPrime(this.inData[j]);
				//this is an optimized version of the last line ↓
				this.costs[j] = err * (1 - Math.pow(this.outData[j],2));
			}
		}

		inSize(){
			return this.inData.length;
		}

		outSize(){
			return this.outData.length;
		}

	}

	swag.TanhLayer = TanhLayer;
	swag.Tanh = TanhLayer;	

}

