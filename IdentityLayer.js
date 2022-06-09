
{

	/*
		Layer specification:

		Each layer needs an inData and outData array

		Each layer needs a "forward" and a "backward" function

		updateParams() updates params from the saved grads from the backward function

		inSize() and outSize() should be implemented
	*/

	class IdentityLayer {
		// this is a linear dummy layer

		constructor(size) {
				this.nextLayer; //the connected layer
				this.inData = new Float64Array(size); //the inData
				this.outData = new Float64Array(size); //will be init when "connect" is called.
				this.costs = new Float64Array(size); //costs for each neuron

		}

		forward(inData){
					if(inData){
						if(inData.length != this.inSize()){
							throw "INPUT SIZE WRONG ON (Input or output or linear) LAYER:\nexpected size ("
							 + this.inSize() + "), got: (" + inData.length + ")";
						}
						for(var i = 0;i<inData.length;i++){
							this.inData[i] = inData[i];
						}						
					}

					for (var h = 0; h < this.outSize(); h++) {
						this.outData[h] = this.inData[h];
					}
		}

		backward(expected){
			if(!expected){
				if(this.nextLayer == undefined){
					throw "nothing to backpropagate!"
				}
				expected = [];
				for(var i = 0;i<this.outData.length;i++){
					this.costs[i] = this.nextLayer.costs[i];
				}
			}else{
				for(var j = 0;j<this.outData.length;j++){
					let err = (expected[j] - this.outData[j]);
					this.costs[j] = err;
				}
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
			//nothing to update, this layer doesnt have parameters
			//Quandale dingle
		}
	}

	swag.IdentityLayer = IdentityLayer;
	swag.Identity = IdentityLayer;
	swag.Input = IdentityLayer;
	swag.DummyLayer = IdentityLayer;
	swag.InputLayer = IdentityLayer;
	swag.OutputLayer = IdentityLayer;
	swag.Output = IdentityLayer;
	

}

