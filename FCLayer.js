
{

	/*
		Layer specification:

		Each layer needs an inData and outData array

		Each layer needs a "forward" and a "backward" function

		updateParams() updates params from the saved grads from the backward function

		inSize() and outSize() should be implemented
	*/

	class FCLayer {
		constructor(inSize, outSize) {

				if(!inSize){
					throw "Missing parameter in FC Constructor: (input size)";
				}

				if(!outSize){
					throw "Missing parameter in FC Constructor: (output size)";
				}
				
				this.ws = new Float64Array(inSize * outSize); //the weights sensitivities to error
				this.bs = new Float64Array(outSize); //the bias sensitivities to error
				this.nextLayer; //the connected layer
				this.inData = new Float64Array(inSize); //the inData
				this.outData = new Float64Array(outSize); 
				this.w = new Float64Array(inSize * outSize); //this will store the weights
				this.b = new Float64Array(outSize); //this will store the biases (biases are for the outData (next layer))
				this.costs = new Float64Array(inSize); //costs for each neuron

				for (var j = 0; j < inSize; j++) {//----init random weights
					for (var h = 0; h < outSize; h++) {
						this.w[h + j * outSize] = (
							1 * Math.random() * (Math.random() > 0.5 ? -1 : 1)
						);
						this.ws[h + j * outSize] = 0;
					}
				} // ---- end init weights

				for (var j = 0; j < outSize; j++) {
					this.b[j] = (
						1 * Math.random() * (Math.random() > 0.5 ? -1 : 1)
					);
					this.bs[j] = 0;
				} ///---------adding random biases
		}

		forward(inData){
					if(inData){
						if(inData.length != this.inSize()){
							throw "INPUT SIZE WRONG ON FC LAYER:\nexpected size ("
							 + this.inSize() + "), got: (" + inData.length + ")";
						}
						for(var i = 0;i<inData.length;i++){
							this.inData[i] = inData[i];
						}
					}


						for (var h = 0; h < this.outSize(); h++) {
							this.outData[h] = 0; //reset outData activation
							for (var j = 0; j < this.inSize(); j++) {
										this.outData[h] +=
											this.inData[j] *
											this.w[h + j * this.outSize()];// the dirty deed
							}
										this.outData[h] += this.b[h];
						}
					
		}

		backward(expected){ 
			for(var i = 0;i<this.inSize();i++){ //reset the costs
				this.costs[i] = 0;
			}

			if(!expected){// -- sometimes the most effiecant way is the least elagant one...
				if(this.nextLayer == undefined){
					throw "error backproping on an unconnected layer with no expected parameter input"
				}
				for(var j = 0;j<this.outSize();j++){
					let err = this.nextLayer.costs[j];
					for(var i = 0;i<this.inSize();i++){
						//activation times error = change to the weight.
						this.ws[j + i * this.outSize()] += 
						this.inData[i] * err * 2;
						this.costs[i] += 
						this.w[j + i * this.outSize()] * err * 2;
					}
					//bias grad is real simple :)
					this.bs[j] += err;
				}
			}else{
				for(var j = 0;j<this.outSize();j++){
					let err = (expected[j] - this.outData[j]);
					for(var i = 0;i<this.inSize();i++){
						//activation times error = change to the weight.
						this.ws[j + i * this.outSize()] += 
						this.inData[i] * err * 2;
						this.costs[i] += 
						this.w[j + i * this.outSize()] * err * 2;
					}
					//bias grad is real simple :)
					this.bs[j] += err;
				}
			}// end of 'sometimes the most effeciant is the least elagant' ----
			/*
				There is no other way... There are other options. Which inlclude makiing
				an error array.. waste of space..... we could just calc err from the 
				expected array??!...... no, we cant. You know that is a waste of cpu, it 
				leads to two USELESS calls of addition. What is wrong with this world. What
				did i ever do to deserve this. Is there any other way? Please someone out there 
				please help me fix the code. This is the most effeicant way there is no other. 
				I cant do anything about it sometimes you have to live with your mistakes. Sometimes 
				your mistakes become masterpeices. The machineries of the nature are similar in the
				way they work. Nature doesnt care if it looks good. Nature only cares about effeciancy.
				Who cares if a little code is copied and pasted it doesnt matter. All that mattters is that
				it saves Two calls of addition. My computer is nature. I will not let traditional views like
				"vanity" sacrifice my codes effeiciancy. 
			*/

			//finish averaging the costs
			for(var i = 0;i<this.inSize();i++){
				this.costs[i] = this.costs[i] / (this.inSize() * this.outSize());
			}
		}

		inSize(){
			return this.inData.length;
		}

		outSize(){
			return this.outData.length;
		}

		updateParams(lr,bs){
			
			//first average out the weights and such

			for(var j = 0;j<this.outSize();j++){
				for(var i = 0;i<this.inSize();i++){
					this.ws[j + i * this.outSize()] /= bs;
					this.w[j + i * this.outSize()] += this.ws[j + i * this.outSize()] * lr;
					this.ws[j + i * this.outSize()] = 0;
					if(this.w[j + i * this.outSize()] == NaN){
						console.log("Ruh roh");
					}
				}
			}

			for(var j = 0;j<this.outSize();j++){
					this.bs[j] /= bs;
					this.b[j] += this.bs[j] * lr;
					this.bs[j] = 0;
			}

		}
	}

	swag.FCLayer = FCLayer;
	swag.FC = FCLayer;
	

}

