
{

	/*
		Layer specification:

		Each layer needs an inData and outData array.
		These arrays should never be replaced as they are shared objects.
		Replacing the inData or outData array will break the net

		Each layer needs a "forward" and a "backward" function

		updateParams() updates params from the saved grads from the backward function

		inSize() and outSize() should be implemented

		OPTIONAL:
			if you want your custom layer to work with mutation/genetic learning add
			"mutate(rate,intensity)" function, mutate your layers parameters

			if you want your layer to have gpu support, add an initGPU function.
			Also add a param bool called 'gpuEnabled' which will turn on when gpu should
			be enabled.
			You can code the rest yourself, swagnet uses gpujs and swag.gpu is a 
			global gpu object to use to prevent making multpile gpu objects.

	*/






	class FCLayer {

		constructor(inSize, outSize) {

				if(!inSize){
					throw "Missing parameter in FC Constructor: (input size)";
				}

				if(!outSize){
					throw "Missing parameter in FC Constructor: (output size)";
				}
				
				this.gpuEnabled = false;
				this.trainIterations = 0; //++'d whenever backwards is called;
				this.ws = new Float32Array(inSize * outSize); //the weights sensitivities to error
				this.bs = new Float32Array(outSize); //the bias sensitivities to error
				this.nextLayer; //the connected layer
				this.inData = new Float32Array(inSize); //the inData
				this.outData = new Float32Array(outSize); 
				this.w = new Float32Array(inSize * outSize); //this will store the weights
				this.b = new Float32Array(outSize); //this will store the biases (biases are for the outData (next layer))
				this.costs = new Float32Array(inSize); //costs for each neuron

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
			this.trainIterations++;
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

		updateParams(lr){
			
			for(var j = 0;j<this.outSize();j++){
				for(var i = 0;i<this.inSize();i++){
					this.ws[j + i * this.outSize()] /= this.trainIterations;
					this.w[j + i * this.outSize()] += this.ws[j + i * this.outSize()] * lr;
					this.ws[j + i * this.outSize()] = 0;
				}
			}

			for(var j = 0;j<this.outSize();j++){
					this.bs[j] /= this.trainIterations;
					this.b[j] += this.bs[j] * lr;
					this.bs[j] = 0;
			}

			this.trainIterations = 0;

		}

		mutate(mutationRate,mutationIntensity){
			for(var i = 0;i<this.w.length;i++){
				if(Math.random() < mutationRate){
					this.w[i] += Math.random() * mutationIntensity * (Math.random() > 0.5 ? -1:1);
				}
			}
			for(var i = 0;i<this.b.length;i++){
				if(Math.random() < mutationRate){
					this.b[i] += Math.random() * mutationIntensity * (Math.random() > 0.5 ? -1:1);
				}
			}
		}



		//----------------
		// GPU specific stuff below! Beware.
		//----------------
	
		// initGPU(){
		// 	this.forwardGPUKernel = swag.gpu.createKernel(function(inData, weights, biases, outDataLength, inDataLength) {
		// 		var act = 0;
		// 		for (var j = 0; j < inDataLength; j++) {
		// 					act +=
		// 						inData[j] *
		// 						weights[this.thread.x + j *	outDataLength];// the dirty deed
		// 		}
		// 		act += biases[this.thread.x];
		// 		return act
		// 	}).setOutput([this.outData.length]);
		// }
	
		//----------------
		// End of GPU specific stuff. Take a breather.
		//----------------
	}



	swag.FCLayer = FCLayer;
	swag.FC = FCLayer;
	

}

