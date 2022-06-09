
{

	/*
		Layer specification:

		Each layer needs an inData and outData array

		Each layer needs a "forward" and a "backward" function

		updateParams() updates params from the saved grads from the backward function

		inSize() and outSize() should be implemented
	*/
	class SigmoidLayer {

		sigmoid(z) {
			return 1 / (1 + Math.exp(-z));
		}
	
		sigmoidPrime(z) {
			return Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2);
		}


		constructor(size) {
				this.nextLayer; //the connected layer
				this.inData = new Float64Array(size); 
				this.outData = new Float64Array(size); 
				this.costs = new Float64Array(size); //costs for each neuron
		}

		forward(inData){
					if(inData){
						if(inData.length != this.inData.length){
							throw "INPUT SIZE WRONG ON SIG LAYER:\nexpected size ("
							 + this.inSize + "), got: (" + inData.length + ")";
						}
						for(var i = 0;i<inData.length;i++){
							this.inData[i] = inData[i];
						}					
					}

					for (var h = 0; h < this.outSize(); h++) {
						this.outData[h] = this.sigmoid(this.inData[h]);
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
				this.costs[j] = err * this.sigmoidPrime(this.inData[j]);
			}
		}

		inSize(){
			return this.inData.length;
		}

		outSize(){
			return this.outData.length;
		}

		updateParams(lr,bs){
			return;
			//In an activation layer, nothing is updated.
			//Quandale dingle
		}
	}

	swag.SigmoidLayer = SigmoidLayer;
	swag.Sigmoid = SigmoidLayer;
	swag.Sig = SigmoidLayer;
	

}

