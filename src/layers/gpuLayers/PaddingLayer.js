{
	class PaddingLayerGPU {
		constructor(inDim, pad, padwith) {
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

			pad = pad || 2;
			padwith = padwith || 0;
			this.pad = pad;
			this.inWidth = inWidth;
			this.inHeight = inHeight;
			this.inDepth = inDepth;
			this.padwith = padwith;
			this.gpuInDataStartIndex = 0;
			this.gpuOutDataStartIndex = 0;
		}

		forward() {
			Ment.webMonkeys.work(
				this.inSize(),
				// prettier-ignore
				`
float act = 0.0;
int remaining = i;
int ifromi = int(remaining / (${this.inHeight * this.inWidth}));
remaining = remaining - (ifromi * ${this.inHeight * this.inWidth});
int jfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (jfromi * ${this.inWidth});
int hfromi = remaining;
int prop = ifromi * ${this.inHeight} * ${this.inWidth} + jfromi * ${this.inWidth} + hfromi;
int index = (jfromi + 1) * ${this.pad} * 2 + -${this.pad} + ${this.pad} * (${this.inWidth} + ${this.pad} * 2) + prop + ifromi * ((${this.inWidth} + ${this.pad} * 2) * ${this.pad}) * 2 + ifromi * (${this.inHeight} * ${this.pad}) * 2;

act = ${this.gpuInDataName}(prop + ${this.gpuInDataStartIndex});

${this.gpuOutDataName}(index + ${this.gpuOutDataStartIndex}) := act;
				`
			);
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
				this.inSize(),
				// prettier-ignore
				`
float act = 0.0;
int remaining = i;
int ifromi = int(remaining / (${this.inHeight * this.inWidth}));
remaining = remaining - (ifromi * ${this.inHeight * this.inWidth});
int jfromi = int(remaining / (${this.inWidth}));
remaining = remaining - (jfromi * ${this.inWidth});
int hfromi = remaining;
int prop = ifromi * ${this.inHeight} * ${this.inWidth} + jfromi * ${this.inWidth} + hfromi;
int index = (jfromi + 1) * ${this.pad} * 2 + -${this.pad} + ${this.pad} * (${this.inWidth} + ${this.pad} * 2) + prop + ifromi * ((${this.inWidth} + ${this.pad} * 2) * ${this.pad}) * 2 + ifromi * (${this.inHeight} * ${this.pad}) * 2;

act = ${this.gpuErrorArrayName}(index);

${this.gpuCostsArrayName}(prop) := act;
				`
			);
		}

		inSize() {
			return this.inWidth * this.inHeight * this.inDepth;
		}

		outSize() {
			return (this.inWidth + this.pad * 2) * (this.inHeight + this.pad * 2) * this.inDepth;
		}

		inSizeDimensions() {
			return [this.inWidth, this.inHeight, this.inDepth];
		}

		outSizeDimensions() {
			return [this.inWidth + this.pad * 2, this.inHeight + this.pad * 2, this.inDepth];
		}

		save() {
			// we cant see the length of arrays after saving them in JSON
			// for some reason so we are adding temp variables so we know
			// the sizes;
			let ret = JSON.stringify(this, function (key, value) {
				//here we define what we need to save
				if (
					key == "inData" ||
					key == "outData" ||
					key == "costs" ||
					key == "nextLayer" ||
					key == "previousLayer" ||
					key == "pl" ||
					key == "gpuInDataName" ||
					key == "gpuOutDataName" ||
					key == "gpuCostsArrayName" ||
					key == "gpuErrorArrayName"
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
			let layer = new PaddingLayerGPU(
				[saveObject.inWidth, saveObject.inHeight, saveObject.inDepth],
				saveObject.pad,
				saveObject.padwith
			);
			return layer;
		}
	}

	Ment.PaddingLayerGPU = PaddingLayerGPU;
	Ment.PadLayerGPU = PaddingLayerGPU;
	Ment.PaddingGPU = PaddingLayerGPU;
	Ment.PadGPU = PaddingLayerGPU;
}
