{
	class Net {
		constructor(layers, minmax) {
			//minmax is the range weights will be in.
			this.layers = [];
			this.batchSize = 19;
			this.epoch = 0;
			this.learningRate = 0.01; //the user can change this whenever theys want
			this.trainingInitialized = false;
			if (minmax === undefined) {
				minmax = 3;
			}
			this.minmax = minmax;
			if (layers !== undefined) {
				this.lastMutations = Array(layers.length);
				//CONNECTING THE LAYERS
				for (var i = 0; i < layers.length; i++) {
					this.lastMutations[i] = {
						wChanges: [],
						wChangesIndex: [],
						bChanges: [],
						bChangesIndex: [],
					};
					this.layers.push(layers[i]);
					var layer = this.layers[i];
					if (i < layers.length - 1) {
						for (var j = 0; j < layer.size; j++) {
							////-------connecting layers
							for (var h = 0; h < layers[i + 1].size; h++) {
								layer.w.push(
									minmax * Math.random() * (Math.random() > 0.5 ? -1 : 1)
								);
							}
						} ///////--------------------------connecting layers
					}
					for (var j = 0; j < layer.size; j++) {
						layer.biases.push(
							minmax * Math.random() * (Math.random() > 0.5 ? -1 : 1)
						);
					} ///---------adding biases
				}
				//END OF CONNECTING THE LAYERS
			}
		} //END OF CONSTRUCTOR

		initTrain() {
			//this has to be called before we train. We need it for optimization
			for (var i = 0; i < this.layers.length; i++) {
				let layer = this.layers[i];
				//add z's
				layer.z = new Float64Array(layer.size);
				//add weight sensitivities
				for (var j = 0; j < layer.w.length; j++) {
					layer.ws.push(0);
				}
				//add bias sensitivities
				layer.bs = new Float64Array(layer.size);
				layer.costs = new Float64Array(layer.size);
			}

			this.trainingInitialized = true;

			this.forward = function (data) {
				/* data is an array of values. I would use typed arrays but they are slower. */
				for (var i = 0; i < this.layers.length; i++) {
					this.layers[i].activations.fill(0); //reset all activations to zero
				}

				if (data.length == this.layers[0].size) {
					//check if sizes are correct
					this.layers[0].activations = [...data]; //does the dirty deed
					for (i = 0; i < this.layers.length; i++) {
						var layer = this.layers[i];
						for (var a = 0; a < layer.activations.length; a++) {
							layer.z[a] = layer.activations[a] + layer.biases[a];
							layer.activations[a] = layer.actFunc(layer.z[a]);
						}
						if (i != this.layers.length - 1) {
							for (var j = 0; j < layer.size; j++) {
								for (var h = 0; h < this.layers[i + 1].size; h++) {
									this.layers[i + 1].activations[h] +=
										layer.activations[j] *
										layer.w[h + j * this.layers[i + 1].size];
								}
							}
						}
					}
				}

				return [...this.layers[this.layers.length - 1].activations]; //return activations of the last layer
			};
		}

		forward(data) {
			/* data is an array of values. I would use typed arrays but they are slower. */
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].activations.fill(0); //reset all activations to zero
			}

			if (data.length == this.layers[0].size) {
				//check if sizes are correct
				this.layers[0].activations = data; //does the dirty deed
				for (i = 0; i < this.layers.length; i++) {
					var layer = this.layers[i];
					for (var a = 0; a < layer.activations.length; a++) {
						layer.activations[a] = layer.actFunc(
							layer.activations[a] + layer.biases[a] //biases zero for first layer
						);
					}
					if (i != this.layers.length - 1) {
						for (var j = 0; j < layer.size; j++) {
							for (var h = 0; h < this.layers[i + 1].size; h++) {
								this.layers[i + 1].activations[h] +=
									layer.activations[j] *
									layer.w[h + j * this.layers[i + 1].size];
							}
						}
					}
				}
			}

			return [...this.layers[this.layers.length - 1].activations]; //return activations of the last layer (copied btw so you can change it if you want)
		}

		mutate(severity, chance) {
			/*  Mutates the weights and biaseses and stuff. For genetic evoloution or whatever :) */
			//severity is how severe the mutations are. chance is how much they occur. ie: 1 chance means all weights are changed. 0 chance no weights are changed.
			for (var i = 0; i < this.layers.length; i++) {
				var layer = this.layers[i];
				this.lastMutations[i] = {
					wChanges: [],
					wChangesIndex: [],
					bChanges: [],
					bChangesIndex: [],
				};
				for (var j = 0; j < layer.w.length; j++) {
					if (Math.random() < chance) {
						let changeAmount =
							Math.random() * severity * (Math.random() < 0.5 ? -1 : 1);
						this.lastMutations[i].wChanges.push(
							-layer.w[j] + swag.bounce(layer.w[j] + changeAmount, this.minmax)
						);
						this.lastMutations[i].wChangesIndex.push(j);
						layer.w[j] = swag.bounce(layer.w[j] + changeAmount, this.minmax);
					}
				}

				if (layer.type !== 'input') {
					//dont change input biases
					for (j = 0; j < layer.biases.length; j++) {
						if (Math.random() < chance) {
							let changeAmount =
								Math.random() * severity * (Math.random() < 0.5 ? -1 : 1);
							this.lastMutations[i].bChanges.push(
								-layer.biases[j] +
									swag.bounce(layer.biases[j] + changeAmount, this.minmax)
							);

							this.lastMutations[i].bChangesIndex.push(j);
							layer.biases[j] = swag.bounce(
								layer.biases[j] + changeAmount,
								this.minmax
							);
						}
					}
				}
			}
		}

		save(saveToFile, filename) {
			//this can be made better.
			//this method returns a string of json used to get the model back
			var savedActivations = [];
			for (var i = 0; i < this.layers.length; i++) {
				//here we take off the activations because they are useless to save.
				savedActivations.push(this.layers[i].activations);
				this.layers[i].activations = undefined;
			}

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
			for (var i = 0; i < this.layers.length; i++) {
				//here we put the activations back on afterwards
				this.layers[i].activations = savedActivations[i];
			}

			return JSON.stringify(this.layers);
		} //end of save method

		load(json) {
			if (typeof json == 'string') {
				json = JSON.parse(json);
			}
			this.layers = []; //reset stuff. were overridding
			for (let i = 0; i < json.length; i++) {
				var layer = new swag.Layer(json[i].type, json[i].size, json[i].afs);
				layer.w = json[i].w;
				layer.biases = json[i].biases;
				this.layers.push(layer);
			}
		} //end of load method

		copy(net) {
			/* This method copies the weights and biaseses of one network into its own weights and biaseses  */
			this.layers = [];
			this.lastMutations = Array(net.layers.length);
			for (var i = 0; i < net.layers.length; i++) {
				var layer = net.layers[i];

				var copiedLayer = new swag.Layer(layer.type, layer.size);
				copiedLayer.actFunc = layer.actFunc;
				copiedLayer.afs = layer.afs;
				copiedLayer.w = layer.w.slice();
				copiedLayer.z = layer.z.slice();
				copiedLayer.biases = layer.biases.slice();
				copiedLayer.activations = new Float64Array(layer.size);

				this.layers.push(copiedLayer);
			}
		} //end of copy method

		unmutate() {
			for (var i = 0; i < this.layers.length; i++) {
				var layer = this.layers[i];
				for (var j = 0; j < this.lastMutations[i].wChangesIndex.length; j++) {
					layer.w[this.lastMutations[i].wChangesIndex[j]] -=
						this.lastMutations[i].wChanges[j];
				}
				for (var j = 0; j < this.lastMutations[i].bChangesIndex.length; j++) {
					layer.biases[this.lastMutations[i].bChangesIndex[j]] -=
						this.lastMutations[i].bChanges[j];
				}
			}
		}
		evolve(data, iterations) {
			var avgLoss = 0;
			for (var i = 0; i < data.length / 2; i++) {
				avgLoss += swag.getLoss(this.forward(data[i * 2]), data[i * 2 + 1]);
			}
			avgLoss = avgLoss / (data.length / 2);
			for (i = 0; i < iterations; i++) {
				this.mutate(this.minmax, 0.5);
				let currentLoss = 0;
				for (var j = 0; j < data.length / 2; j++) {
					currentLoss += swag.getLoss(
						this.forward(data[j * 2]),
						data[j * 2 + 1]
					);
				}
				currentLoss = currentLoss / (data.length / 2);
				if (currentLoss <= avgLoss) {
					avgLoss = currentLoss;
				} else {
					this.unmutate();
				}
			}
			return avgLoss;
		}

		train(input, expectedOut) {
		  
		  //error with cost back prop. It works with one layer but not two.
		  
			//do last layer first
			
			this.epoch++;

			if (!this.trainingInitialized) {
				this.initTrain();
			}
			this.forward(input);

			for (var i = 0; i < this.layers[this.layers.length - 1].size; i++) {
				let layer = this.layers[this.layers.length - 1];
				layer.costs[i] = layer.activations[i] - expectedOut[i]; //we will power^2 at the end.
			}

			for (i = this.layers.length - 2; i >= 0; i--) {
				let layer = this.layers[i];
				let layerNext = this.layers[i + 1];
				for (var j = 0; j < layer.size; j++) {
					for (var h = 0; h < layerNext.size; h++) {

						layer.ws[h + j * layerNext.size] +=
							layer.activations[j] *
							layerNext.actFuncPrime(layerNext.z[h]) *
							2 *
							layerNext.costs[h];

						layer.costs[j] = 0;
						layer.costs[j] +=
							(layer.w[h + j * layerNext.size] *
							layerNext.actFuncPrime(layerNext.z[h]) *
							2 *
							layerNext.costs[h]);
					}
				 	//layer.costs[j] = layer.activations[j] + layer.costs[j];

				}
				for (h = 0; h < layerNext.size; h++) {
					layerNext.bs[h] +=
						layerNext.actFuncPrime(layerNext.z[h]) * 2 * layerNext.costs[h];
				}
			}

			if (this.epoch >= this.batchSize) {
				//apply sensitivities for every weights and bias
				for (i = this.layers.length - 1; i >= 0; i--) {
					var layer = this.layers[i];
					for (j = 0; j < layer.ws.length; j++) {
						layer.w[j] += -layer.ws[j]/this.batchSize * this.learningRate;
						layer.ws[j] = 0;
					}

					for (j = 0; j < layer.bs.length; j++) {
						layer.biases[j] += -layer.bs[j]/this.batchSize * this.learningRate;
						layer.bs[j] = 0;
					}

					for (j = 0; j < layer.costs.length; j++) {
						layer.costs[j] = 0;
					}
				}
				this.epoch = 0;
			}

			for (i = 0; i < this.layers[this.layers.length - 1].size; i++) {
				let layer = this.layers[this.layers.length - 1];
				layer.costs[i] = Math.pow(layer.costs[i], 2); //we will power^2 at the end.
			}


			return JSON.stringify(this.layers[this.layers.length - 1].costs[0]);
		}
	} //END OF NET CLASS DECLARATION

	swag.Net = Net;
}
