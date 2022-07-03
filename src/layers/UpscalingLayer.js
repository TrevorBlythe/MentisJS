{
	class UpscalingLayer {
		constructor(inDim, scale) {
			if(inDim.length != 3){
				throw this.constructor.name + " parameter error: Missing dimensions parameter. \n"
				+ "First parameter in layer must be an 3 length array, width height and depth";
			}
			let inWidth = inDim[0];
			let inHeight = inDim[1];
			let inDepth = inDim[2];
			
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.outWidth = inWidth * scale;
			this.outHeight = inHeight * scale;
			this.scale = scale;
			this.nextLayer; //the connected layer
			this.previousLayer; //the previousLayer
			this.inData = new Float32Array(inWidth * inHeight * inDepth); //the inData
			this.outData = new Float32Array(this.outWidth * this.outHeight * inDepth);
			this.costs = new Float32Array(this.inData.length); //costs for each activation in "inData"
		}
		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [
				this.outWidth,this.outHeight,this.inDepth
			];
		}
		forward(inData) {
			if (inData) {
				if (inData.length != this.inSize()) {
					throw inputError(this, inData);
				}
				//Fun fact: the fastest way to copy an array in javascript is to use a For Loop
				for (var i = 0; i < inData.length; i++) {
					this.inData[i] = inData[i];
				}
			}

			for(var i = 0;i<this.inDepth;i++){
				for(var h = 0;h<this.inHeight * this.scale;h++){
					for(var j = 0;j<this.inWidth * this.scale;j++){
						this.outData[(i * this.outHeight * this.outWidth) + (h * this.outWidth) + j]
						= 
						this.inData[(Math.floor((i/this.scale)) * this.inHeight * this.inWidth) + (Math.floor((h/this.scale)) * this.inWidth) + Math.floor(j/this.scale)];
					}
				}
			}
		}

		backward(expected) {
			let loss = 0;
			this.costs.fill(0);
			let geterr = (ind) => {
				return expected[ind] - this.outData[ind];
			};
			if (!expected) {
				geterr = (ind) => {
					return this.nextLayer.costs[ind];
				};
				if (this.nextLayer == undefined) {
					throw 'error backproping on an unconnected layer with no expected parameter input';
				}
			}

			for(var i = 0;i<this.inDepth;i++){
				for(var h = 0;h<this.inHeight * this.scale;h++){
					for(var j = 0;j<this.inWidth * this.scale;j++){
						this.costs[(Math.floor((i/this.scale)) * this.inHeight * this.inWidth) + (Math.floor((h/this.scale)) * this.inWidth) + Math.floor(j/this.scale)]
						+= 
						geterr((i * this.outHeight * this.outWidth) + (h * this.outWidth) + j);
						loss += geterr((i * this.outHeight * this.outWidth) + (h * this.outWidth) + j);
					}
				}
			}
			// for(var i = 0;i<this.costs.length;i++){
			// 	this.costs[i] /= this.scale;
			// }

			return loss/this.outSize();
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
				if (
					key == 'ws' ||
					key == 'bs' ||
					key == 'inData' ||
					key == 'outData' ||
					key == 'costs' ||
					key == 'gpuEnabled' ||
					key == 'trainIterations' ||
					key == 'nextLayer' ||
					key == 'previousLayer' ||
					key == 'pl'
				) {
					return undefined;
				}

				return value;
			});


			return ret;
		}
		//hey if your enjoying my library contact me trevorblythe82@gmail.com

		static load(json) {
			let saveObject = JSON.parse(json);
			let layer = new UpscalingLayer([saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],saveObject.scale);
			return layer;
		}

	}

	Ment.UpscalingLayer = UpscalingLayer;
	Ment.Upscaling = UpscalingLayer;
	Ment.Upscale = UpscalingLayer;
	Ment.UpScale = UpscalingLayer;
	Ment.UpScaling = UpscalingLayer;
	Ment.UpScalingLayer = UpscalingLayer;
}
