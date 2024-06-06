{
	class DepaddingLayerGPU {
		constructor(outDim, pad) {
			if (outDim.length != 3) {
				throw (
					this.constructor.name +
					" parameter error: Missing dimensions parameter. \n" +
					"First parameter in layer must be an 3 length array, width height and depth"
				);
			}
			let outWidth = outDim[0];
			let outHeight = outDim[1];
			let outDepth = outDim[2];

			pad = pad || 2;
			this.pad = pad;
			this.outWidth = outWidth;
			this.outHeight = outHeight;
			this.outDepth = outDepth;
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
		}

		forward() {
			// Ment.webMonkeys.fill(this.gpuOutDataName, this.padwith);
			Ment.webMonkeys.work(
				this.outSize(),
				// prettier-ignore
				`
float act = 0.0;
int remaining = i;
int ifromi = int(remaining / (${this.outHeight * this.outWidth}));
remaining = remaining - (ifromi * ${this.outHeight * this.outWidth});
int jfromi = int(remaining / (${this.outWidth}));
remaining = remaining - (jfromi * ${this.outWidth});
int hfromi = remaining;
int prop = ifromi * ${this.outHeight} * ${this.outWidth} + jfromi * ${this.outWidth} + hfromi;
int index = (jfromi + 1) * ${this.pad} * 2 + -${this.pad} + ${this.pad} * (${this.outWidth} + ${this.pad} * 2) + prop + ifromi * ((${this.outWidth} + ${this.pad} * 2) * ${this.pad}) * 2 + ifromi * (${this.outHeight} * ${this.pad}) * 2;

act = ${this.gpuInDataName}(index + ${this.gpuInDataStartIndex});

${this.gpuOutDataName}(prop + ${this.gpuOutDataStartIndex}) := act;
				`
			);
		}
		inSize() {
			return (this.outWidth + this.pad * 2) * (this.outHeight + this.pad * 2) * this.outDepth;
		}

		outSize() {
			return this.outWidth * this.outHeight * this.outDepth;
		}

		outSizeDimensions() {
			return [this.outWidth, this.outHeight, this.outDepth];
		}

		inSizeDimensions() {
			return [this.outWidth + this.pad * 2, this.outHeight + this.pad * 2, this.outDepth];
		}

		get outData() {
			return Ment.webMonkeys
				.get(this.gpuOutDataName)
				.slice(this.gpuOutDataStartIndex, this.gpuOutDataStartIndex + this.outSize());
		}

		get inData() {
			return Ment.webMonkeys.get(this.gpuInDataName).slice(this.gpuInDataStartIndex, this.gpuInDataStartIndex + this.inSize());
		}

		backward(err) {
			if (err && err.length == this.outSize()) {
				Ment.webMonkeys.set(this.gpuErrorArrayName, err);
			}
			Ment.webMonkeys.work(
				this.outSize(),
				// prettier-ignore
				`
float act = 0.0;
int remaining = i;
int ifromi = int(remaining / (${this.outHeight * this.outWidth}));
remaining = remaining - (ifromi * ${this.outHeight * this.outWidth});
int jfromi = int(remaining / (${this.outWidth}));
remaining = remaining - (jfromi * ${this.outWidth});
int hfromi = remaining;
int prop = ifromi * ${this.outHeight} * ${this.outWidth} + jfromi * ${this.outWidth} + hfromi;
int index = (jfromi + 1) * ${this.pad} * 2 + -${this.pad} + ${this.pad} * (${this.outWidth} + ${this.pad} * 2) + prop + ifromi * ((${this.outWidth} + ${this.pad} * 2) * ${this.pad}) * 2 + ifromi * (${this.outHeight} * ${this.pad}) * 2;

act = ${this.gpuErrorArrayName}(prop);

${this.gpuGradsArrayName}(index) := act;
				`
			);
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "netObject" ||
					key == "outData" ||
					key == "grads" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuGradsArrayName" ||
					key == "gpuErrorArrayName" ||
					key == "accessed"
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
			let layer = new DepaddingLayerGPU([saveObject.outWidth, saveObject.outHeight, saveObject.outDepth], saveObject.pad);
			return layer;
		}
	}

	Ment.DePaddingLayerGPU = DepaddingLayerGPU;
	Ment.DepaddingLayerGPU = DepaddingLayerGPU;
	Ment.DepadLayerGPU = DepaddingLayerGPU;
	Ment.DepaddingGPU = DepaddingLayerGPU;
	Ment.DepadGPU = DepaddingLayerGPU;
}
