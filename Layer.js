
{
  
 
 
 
  
	class Layer {
		constructor(type, size, actFunc) {
			this.type = type; //fc conv whatevs
			this.size = size; //neuron amount
			this.costs = [];
			this.ws = []; //the sensitivities
			this.bs = []; //the bias sensitivities
			this.z = []; //this is just weighted sum of each neuron before the activation function. for backpropping.
			this.afs = actFunc; //this is just stored for saving/loading models
			this.activations = new Float64Array(size); //the activations
			this.w = []; //this will store the weights
			this.biases = []; //this will store the biaseses
			
			if (typeof actFunc == 'string') {
				eval('this.actFunc = swag.' + actFunc);
				eval('this.actFuncPrime = swag.' + actFunc + 'Prime');
				this.afs = actFunc;
			}
			if (actFunc === undefined) {
				this.afs = 'lin';
				this.actFunc = function (num) {
					return num;
				};
			}
		}
	}

	swag.Layer = Layer;
	

}

