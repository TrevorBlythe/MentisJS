
{


	// [(W−K+2P)/S]+1.

	// W is the input volume - in your case 128
	// K is the Kernel size - in your case 5
	// P is the padding - in your case 0 i believe
	// S is the stride - which you have not provided.


//filters stored as such
//this.filters[] array of num
/*
if filter is 3x3 with indepth of 3 the first filter stored as such:
this.filterw = [0,1,2,3,4,5,6,7,8,
								0,1,2,3,4,5,6,7,8,
								0,1,2,3,4,5,6,7,8]

Each filter has depth equal to the InDepth, thats why there are three 3x3 in the array.

this.filterw
[filterNum * this.inDepth * filterWidth * filterHeight + x + (y * filterWidth) + (depth * filterWidth * filterHeight)]

*/



	class ConvLayer {

		constructor(inWidth, inHeight, inDepth, filterWidth, filterHeight, filters=3, stride=1, padding=0) {

				this.filters = filters;//the amount of filters
				this.inWidth = inWidth;
				this.inHeight = inHeight;
				this.inDepth = inDepth;
				this.filterWidth = filterWidth;
				this.filterHeight = filterHeight;
				this.stride = stride;
				this.padding = padding;
				this.filterw = new Float32Array(filters * inDepth * filterWidth * filterHeight);
				this.filterws = new Float32Array(filters * inDepth * filterWidth * filterHeight);
				this.trainIterations = 0; 
				this.outData = new Float32Array(((inWidth-filterWidth+2*padding)/stride+1) * ((inWidth-filterWidth+2*padding)/stride+1) * this.filters);
				this.inData = new Float32Array(inWidth * inHeight * inDepth);
				this.costs = new Float32Array(inWidth * inHeight * inDepth);
				this.b = new Float32Array(this.outData.length);
				this.bs = new Float32Array(this.outData.length);
				this.accessed = new Float32Array(this.inData.length);
				this.waccessed = new Float32Array(this.filterw.length);
				if(this.filterWidth > inWidth + padding || this.filterHeight > inHeight + padding){
					throw "Conv layer error: filters cannot be bigger than the input";
				}
				if((this.inWidth / stride) % 1 != 0 || (this.inHeight / stride) % 1 != 0){
					console.log("WARNING: stride parameter should be a factor of input width and input height. or else it might not work properly")
				}
				//init random weights
				for(var i = 0;i<this.filterw.length;i++){
					this.filterw[i] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
				for(var i = 9;i<18;i++){
					this.filterw[i] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
				for(var i = 0;i<this.b.length;i++){
					this.b[i] = 1 * Math.random() * (Math.random() > 0.5 ? -1 : 1);
				}
		}

		inSize(){
			return this.inData.length;
		}

		outSize(){
			return this.outData.length;
		}

		forward(inData){
					if(inData){
						if(inData.length != this.inSize()){
							throw "INPUT SIZE WRONG ON CONV LAYER:\nexpected size ("
							 + this.inSize() + "), got: (" + inData.length + ")";
						}
						for(var i = 0;i<inData.length;i++){
							this.inData[i] = inData[i];
						}
					}
					this.outData.fill(0);
	
					for(var i = 0;i<this.filters;i++){
						for(var g = 0;g<this.inHeight - this.filterHeight + 1;g++){
							for(var b = 0;b<this.inWidth - this.filterWidth + 1;b++){
								let odi = b + (g * (this.inWidth - this.filterWidth + 1)) + (i * (this.inWidth - this.filterWidth + 1) * (this.inHeight - this.filterHeight + 1));
								for(var h = 0;h<this.inDepth;h++){
									for(var j = 0;j<this.filterHeight;j++){
										for(var k = 0;k<this.filterWidth;k++){
											this.outData[odi] += 
											this.inData[(b+k) + ((j+g)*this.inWidth) + (h * this.inWidth * this.inHeight)] * 
											this.filterw[i * this.inDepth * this.filterWidth * this.filterHeight + k + (j * this.filterWidth) + (h * this.filterWidth * this.filterHeight)];
										}
									}
								}
								// this.outData[b + (g * (this.inWidth - this.filterWidth + 1)) + (i * (this.inWidth - this.filterWidth + 1) * (this.inHeight - this.filterHeight + 1))] /= this.inDepth * this.filterHeight * this.filterWidth;
								//pick one?
								// this.outData[b + (g * (this.inWidth - this.filterWidth + 1)) + (i * (this.inWidth - this.filterWidth + 1) * (this.inHeight - this.filterHeight + 1))] /= this.inDepth
								this.outData[odi] += this.b[odi];
							}
						}
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
				
				for(var i = 0;i<this.filters;i++){
					for(var g = 0;g<this.inHeight - this.filterHeight + 1;g++){
						for(var b = 0;b<this.inWidth - this.filterWidth + 1;b++){
							let odi = b + (g * (this.inWidth - this.filterWidth + 1)) + (i * (this.inWidth - this.filterWidth + 1) * (this.inHeight - this.filterHeight + 1));
							let err = this.nextLayer.costs[odi];
							for(var h = 0;h<this.inDepth;h++){
								for(var j = 0;j<this.filterHeight;j++){
									for(var k = 0;k<this.filterWidth;k++){
										this.costs[(b+k) + ((j+g)*this.inWidth) + (h * this.inWidth * this.inHeight)] +=
										this.filterw[i * this.inDepth * this.filterWidth * this.filterHeight + k + (j * this.filterWidth) + (h * this.filterWidth * this.filterHeight)] * err;
										this.waccessed[i * this.inDepth * this.filterWidth * this.filterHeight + k + (j * this.filterWidth) + (h * this.filterWidth * this.filterHeight)] += 1;
										this.accessed[(b+k) + ((j+g)*this.inWidth) + (h * this.inWidth * this.inHeight)] += 1;
										this.filterws[i * this.inDepth * this.filterWidth * this.filterHeight + k + (j * this.filterWidth) + (h * this.filterWidth * this.filterHeight)] += 
										this.inData[(b+k) + ((j+g)*this.inWidth) + (h * this.inWidth * this.inHeight)] * err;
									}
								}
							}
							this.bs[odi] += err;
						}
					}
				}

			}else{

			
				for(var i = 0;i<this.filters;i++){
					for(var g = 0;g<this.inHeight - this.filterHeight + 1;g++){
						for(var b = 0;b<this.inWidth - this.filterWidth + 1;b++){
							let odi = b + (g * (this.inWidth - this.filterWidth + 1)) + (i * (this.inWidth - this.filterWidth + 1) * (this.inHeight - this.filterHeight + 1));
							let err = (expected[odi] - this.outData[odi]);
							for(var h = 0;h<this.inDepth;h++){
								for(var j = 0;j<this.filterHeight;j++){
									for(var k = 0;k<this.filterWidth;k++){
										this.costs[(b+k) + ((j+g)*this.inWidth) + (h * this.inWidth * this.inHeight)] +=
										this.filterw[i * this.inDepth * this.filterWidth * this.filterHeight + k + (j * this.filterWidth) + (h * this.filterWidth * this.filterHeight)] * err;
										this.waccessed[i * this.inDepth * this.filterWidth * this.filterHeight + k + (j * this.filterWidth) + (h * this.filterWidth * this.filterHeight)] += 1;
										this.accessed[(b+k) + ((j+g)*this.inWidth) + (h * this.inWidth * this.inHeight)] += 1;
										this.filterws[i * this.inDepth * this.filterWidth * this.filterHeight + k + (j * this.filterWidth) + (h * this.filterWidth * this.filterHeight)] += 
										this.inData[(b+k) + ((j+g)*this.inWidth) + (h * this.inWidth * this.inHeight)] * err;
									}
								}
							}
							this.bs[odi] += err;
						}
					}
				}

			}// end of 'sometimes the most effeciant is the least elagant' ----

			for(var i = 0;i<this.inSize();i++){
				this.costs[i] = this.costs[i] / this.accessed[i];
				this.accessed[i] = 0;
			}
		}

		updateParams(lr){
			console.log('updating params');
			for(var i = 0 ; i < this.filterw.length; i++){
				this.filterws[i] /=  this.waccessed[i];
				this.waccessed[i] = 0;
				this.filterw[i] += this.filterws[i] * lr;
				this.trainIterations = 0;
			}
		}

	}



	swag.ConvLayer = ConvLayer;
	swag.Conv = ConvLayer;
	

}

