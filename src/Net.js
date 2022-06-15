
var swag = swag || {};
{

	class Net {
		constructor(layers) {
			this.layers = [];
			this.batchSize = 19;
			this.epoch = 0;
			this.learningRate = 0.01; //the user can change this whenever theys want

			if (layers !== undefined) {
				//CONNECTING THE LAYERS
				for (var i = 0;i<layers.length-1;i++) {
					this.layers.push(layers[i]);

					if(layers[i].outSize() != layers[i+1].inSize()){
						throw "Failure connecting " + layers[i].constructor.name + " layer with " + layers[i+1].constructor.name + ","
						+ layers[i].constructor.name + " output size: " + layers[i].outSize() + ", " + layers[i+1].constructor.name + " input size: "
						+ layers[i+1].inSize();
					}
					layers[i+1].inData = layers[i].outData;
					layers[i].nextLayer = layers[i+1];
				}
				this.layers.push(layers[layers.length - 1]);
				//END OF CONNECTING THE LAYERS
			}
		} //END OF CONSTRUCTOR

		forward(data) {
			if(!Array.isArray(data)){
				if(typeof data == 'number'){
					data = [data]
				}else{
					throw "INPUT ARRAYS INTO FORWARDS FUNCTION ONLY!"
				}
			}
			this.layers[0].forward(data);
			for(var i = 1;i<this.layers.length;i++){
				this.layers[i].forward();
			}
			return new Float32Array(this.layers[this.layers.length - 1].outData); //return inData of the last layer (copied btw so you can change it if you want)
		}

		enableGPU(){
			for(var i = 0;i<this.layers.length;i++){
				this.layers[i].gpuEnabled = true;
				if(this.layers[i].initGPU){
					this.layers[i].initGPU()
				}
			}
		}

		mutate(rate,intensity){
			for(var i = 0;i<this.layers.length;i++){
				if(this.layers[i].mutate){
					this.layers[i].mutate(rate,intensity)
				}
			}
		}

		disableGPU(){
			for(var i = 0;i<this.layers.length;i++){
				this.layers[i].gpuEnabled = false;
			}
		}

		save(saveToFile, filename) {
			//this can be made better.
			//this method returns a string of json used to get the model back

			if (saveToFile) {
				if (swag.isBrowser()) {
					var a = document.createElement('a');
					var file = new Blob([JSON.stringify(this.layers)]);
					a.href = URL.createObjectURL(file);
					a.download = 'model.json';
					a.click();
				} else {
					var fs = require('fs');
					// console.log(filename);
					fs.writeFileSync(filename, JSON.stringify(this.layers));
				}
			}
			return JSON.stringify(this.layers);
		} //end of save method

		load(json) {

		} //end of load method

		copy(net) {
			/* This method copies the weights and bes of one network into its own weights and bes  */

		} //end of copy method

		train(input, expectedOut) {
		  this.forward(input);
			let loss = 0;
			for(var i = 0;i<this.layers[this.layers.length-1].outData.length;i++){
				//inefficiant, but more modular than the other option and its not that
				//much of a computation unless you are outputting a lot of things.
				loss += Math.pow(expectedOut[i] - this.layers[this.layers.length-1].outData[i],2);
			}
			loss /= this.layers[this.layers.length-1].outData.length;
			this.layers[this.layers.length - 1].backward(expectedOut);
			for (var i = this.layers.length - 2; i >= 0; i--) {
				this.layers[i].backward();
			}
			this.epoch++;
			if(this.epoch % this.batchSize == 0){
				for (var i = this.layers.length - 1; i >= 0; i--) {
					if(this.layers[i].updateParams)
					this.layers[i].updateParams(this.learningRate);
				}
			}
			return loss;
		}
	} //END OF NET CLASS DECLARATION

	swag.Net = Net;
}
