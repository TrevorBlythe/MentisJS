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
			this.layers[0].forward(data);
			for(var i = 1;i<this.layers.length;i++){
				this.layers[i].forward();
			}
			return [...this.layers[this.layers.length - 1].outData]; //return inData of the last layer (copied btw so you can change it if you want)
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
			this.layers[this.layers.length - 1].backward(expectedOut);
			for (var i = this.layers.length - 2; i >= 0; i--) {
				this.layers[i].backward();
			}
			this.epoch++;
			if(this.epoch % this.batchSize == 0){
				for (var i = this.layers.length - 1; i >= 0; i--) {
					this.layers[i].updateParams(this.learningRate,this.batchSize);
				}
			}
		}
	} //END OF NET CLASS DECLARATION

	swag.Net = Net;
}
