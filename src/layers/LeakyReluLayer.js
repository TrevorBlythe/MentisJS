{
	class LeakyReluLayer extends Ment.ActivationBase {
		constructor(customSlope, size) {
			if (customSlope > 1) {
				console.log(
					"WARNING: the leaky relu constructor parameters have been changed. The first parameter is now the slope of the negative side of the function, which should be around 0.1. The second parameter is now the size of the layer. The old constructor parameters were (size, customSlope). Please update your code. Sorry for the inconvenience. To silence this warning, delete this line of code. And by the way, you dont even have to provide the size parameter anymore, it gets set automaticaly when the layer is connected, unless this is the very first layer in the network."
				);
			}
			super(size); //size gets set automaticaly when the layer is connected.
			this.customSlope = customSlope || 0.1;
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

			for (var h = 0; h < this.outSize(); h++) {
				this.outData[h] = this.inData[h] > 0 ? this.inData[h] : this.inData[h] * this.customSlope;
			}
		}

		backward(err) {
			if (!err) {
				err = this.nextLayer.grads;
			}

			for (var j = 0; j < this.outSize(); j++) {
				if (this.outData[j] >= 0) {
					this.grads[j] = err[j];
				} else {
					this.grads[j] = err[j] * this.customSlope;
				}
			}
		}

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new LeakyReluLayer(saveObject.savedSize, saveObject.customSlope);
			return layer;
		}
	}
	Ment.LeakyReluLayer = LeakyReluLayer;
	Ment.LeakyRelu = LeakyReluLayer;
	Ment.LRelu = LeakyReluLayer;
}
