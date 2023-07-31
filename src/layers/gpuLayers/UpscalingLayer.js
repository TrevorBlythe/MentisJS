{
	class UpscalingLayerGPU {
		constructor(inDim, scale) {
			if (inDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
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
		}
		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [this.outWidth, this.outHeight, this.inDepth];
		}
		forward() {
			Ment.webMonkeys.work(
				this.outSize(),
				`
float act = 0.0;

int remaining = i;
int ifromi = int(remaining / (${this.inHeight * this.scale * this.inWidth * this.scale}));
remaining = remaining - (ifromi * ${this.scale * this.inHeight * this.inWidth * this.scale});
int hfromi = int(remaining / (${this.inWidth * this.scale}));
remaining = remaining - (hfromi * ${this.inWidth * this.scale});
int jfromi = remaining;

act += ${this.gpuInDataName}(${this.gpuInDataStartIndex} + ifromi * ${this.inHeight * this.inWidth} + int(hfromi / ${
					this.scale
				}) * ${this.inWidth} + int(jfromi / ${this.scale}))

;

${this.gpuOutDataName}(i + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}
			Ment.webMonkeys.work(
				this.inSize(),
				`
float act = 0.0;

int remaining = i;
int ifromi = int(remaining / (${this.inHeight * this.inWidth}));
remaining = remaining - (ifromi * ${this.inHeight * this.inWidth});
int hfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (hfromi * ${this.inWidth});
int jfromi = remaining;

for(int c = 0; c < ${this.scale}; c++){
for(int b = 0; b < ${this.scale}; b++){

act += ${this.gpuErrorArrayName}(ifromi * ${this.outHeight * this.outWidth} + (((hfromi * ${this.scale}) + c) * ${
					this.outWidth
				}) + (jfromi * ${this.scale} + b));

};
};

${this.gpuCostsArrayName}(i) := act;
				`
			);
		}

		inSize() {
			return this.inHeight * this.inWidth * this.inDepth;
		}

		outSize() {
			return this.outHeight * this.outWidth * this.inDepth;
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		save() {
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "ws" ||
					key == "bs" ||
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "gpuEnabled" ||
					key == "trainIterations" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl"
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
			let layer = new UpscalingLayerGPU([saveObject.inWidth, saveObject.inHeight, saveObject.inDepth], saveObject.scale);
			return layer;
		}
	}

	Ment.UpscalingLayerGPU = UpscalingLayerGPU;
	Ment.UpscalingGPU = UpscalingLayerGPU;
	Ment.UpscaleGPU = UpscalingLayerGPU;
	Ment.UpScalingGPU = UpscalingLayerGPU;
	Ment.UpscalingLayerGPU = UpscalingLayerGPU;
}
