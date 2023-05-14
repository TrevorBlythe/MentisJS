Ment.polluteGlobal();

var drawFromArr = function (arr, ctx, width, height, scale) {
	for (var i = 0; i < height; i++) {
		for (var j = 0; j < width; j++) {
			let r = arr[i * width + j] * 255;
			let g = arr[i * width + j + width * height] * 255;
			let b = arr[i * width + j + width * height * 2] * 255;
			ctx.fillStyle = "rgb(" + r + "," + g + "," + b + ")";
			ctx.fillRect(j * scale, i * scale, scale, scale);
		}
	}
};

//CONV LAYER CODE
var cLayer = new ConvLayer([6, 6, 3], [2, 2], 3, 1, false); //filter size 4 by 4, 2 filters, stride of 1, no bias

//manually set weights cuz random weights suck
// prettier-ignore
cLayer.filterw = [
	1,0,
	1,0,

	0,0,
	0,0,
	
	0,0,
	0,0,
//end of first filter

	0,0,
	0,0,

	0,0,
	0,0,

	0,0,
	0,0,
//end of second filter

	0,0,
	0,0,

	0,0,
	0,0,

	0,0,
	0,0
//end of third filter
];

// prettier-ignore
var cLayerInput = [
	//red pixel values
	0, 0, 0, 0, 0, 0, 
	0, 1, 0, 1, 0, 0, 
	0, 1, 0, 1, 0, 0, 
	0, 1, 0, 1, 0, 0, 
	0, 1, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0,

	//green pixel values
	0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 1, 0, 
	0, 0, 1, 0, 1, 0, 
	0, 0, 1, 0, 1, 0, 
	0, 0, 1, 0, 1, 0, 
	0, 0, 0, 0, 0, 0,

	//blue pixel values
	0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0
];

cLayer.forward(cLayerInput);

//convCanvWeights
var convCanvIn = document.getElementById("convCanvIn").getContext("2d");
var convCanvOut = document.getElementById("convCanvOut").getContext("2d");
drawFromArr(cLayerInput, convCanvIn, 6, 6, 5);
drawFromArr(cLayer.outData, convCanvOut, 5, 5, 5);

var convCanvWeights = document.getElementById("convCanvWeights").getContext("2d");
var convCanvWeightsTwo = document.getElementById("convCanvWeightsTwo").getContext("2d");
var convCanvWeightsThree = document.getElementById("convCanvWeightsThree").getContext("2d");
drawFromArr(cLayer.filterw, convCanvWeights, 2, 2, 5);
drawFromArr(cLayer.filterw.slice(2 * 2 * 3, 2 * 2 * 3 * 2), convCanvWeightsTwo, 2, 2, 5);
drawFromArr(cLayer.filterw.slice(2 * 2 * 3 * 2, 2 * 2 * 3 * 3), convCanvWeightsThree, 2, 2, 5);

// DECONV LAYER CODE-----------------------------------------------------------------------------------------------------------
//for deconv layers the dimensions you put in are what it will output
var dCLayer = new DeConvLayer([6, 6, 3], [2, 2], 3, 1, false); //filter size 4 by 4, 2 filters, stride of 1, no bias

//manually set weights cuz random weights suck
// prettier-ignore
dCLayer.filterw = [
	0.5,0,
	0.5,0,

	0,0,
	0,0,

	0,0,
	0,0,
//end of first filter

	0,0,
	0,0,

	0,0,
	0,0,

	0,0,
	0,0,
//end of second filter

	0,0,
	0,0,

	0,0,
	0,0,

	0,0,
	0,0
//end of third filter
];

// prettier-ignore
var dCLayerInput = [
	0,1,0,1,0,
	0,1,0,1,0,
	0,1,0,1,0,
	0,1,0,1,0,
	0,1,0,1,0,

	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,

	1,0,0,0,1,
	1,0,0,0,1,
	1,0,0,0,1,
	1,0,0,0,1,
	1,0,0,0,1,
];

dCLayer.forward(dCLayerInput);

//convCanvWeights
var deConvCanvIn = document.getElementById("deConvCanvIn").getContext("2d");
var deConvCanvOut = document.getElementById("deConvCanvOut").getContext("2d");
drawFromArr(dCLayerInput, deConvCanvIn, 5, 5, 5);
drawFromArr(dCLayer.outData, deConvCanvOut, 6, 6, 5);

var deConvCanvWeights = document.getElementById("deConvCanvWeights").getContext("2d");
var deConvCanvWeightsTwo = document.getElementById("deConvCanvWeightsTwo").getContext("2d");
var deConvCanvWeightsThree = document.getElementById("deConvCanvWeightsThree").getContext("2d");
drawFromArr(dCLayer.filterw, deConvCanvWeights, 2, 2, 5);
drawFromArr(dCLayer.filterw.slice(2 * 2 * 3, 2 * 2 * 3 * 2), deConvCanvWeightsTwo, 2, 2, 5);
drawFromArr(dCLayer.filterw.slice(2 * 2 * 3 * 2, 2 * 2 * 3 * 3), deConvCanvWeightsThree, 2, 2, 5);

//DENSE LAYER CODE FULLY CONNECTED LAYER CODE -----------------------------------------------------------
// prettier-ignore
var fcLayer = new Net([
	new FCLayer(2, 3),
	new FCLayer(3,2),]); //wrap it in a Net to be rendered
fcLayer.forward([0, 0.5]);
var ctxN = document.getElementById("fcCanv").getContext("2d");
render({
	net: fcLayer,
	ctx: ctxN,
	x: 10,
	y: 10,
	scale: 16,
	background: "pink",
	spread: 60,
});

//PADDING LAYER CODE -----------------------------------
var pLayer = new PaddingLayer([6, 6, 3], 2, 0); //2 pixels of padding on each side, set the padding to 0
var dPLayer = new DePaddingLayer([6, 6, 3], 2); //Take away 2 pixels of padding each side to make a [6,6,3] image

// prettier-ignore
var pLayerInput = [
	//red pixel values
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1,

	//green pixel values
	1, 1, 1, 1, 1, 1, 
	1, 0, 0, 0, 0, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1,

	//blue pixel values
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 
	1, 0, 0, 0, 0, 1, 
	1, 1, 1, 1, 1, 1
];

pLayer.forward(pLayerInput);

var padIn = document.getElementById("padIn").getContext("2d");
var padOut = document.getElementById("padOut").getContext("2d");
drawFromArr(pLayerInput, padIn, 6, 6, 3);
drawFromArr(pLayer.outData, padOut, 10, 10, 3);

dPLayer.forward(pLayer.outData);
var dePadIn = document.getElementById("dePadIn").getContext("2d");
var dePadOut = document.getElementById("dePadOut").getContext("2d");
drawFromArr(pLayer.outData, dePadIn, 10, 10, 3);
drawFromArr(dPLayer.outData, dePadOut, 6, 6, 3);

//UPSCALING LAYER CODE----------------------------------
var upLayer = new Upscale([2, 2, 3], 2);
var upInput = [
	1, 1, 0, 0,

	0, 0, 0, 0,

	0, 0, 1, 1,
	//end of third filter
];
upLayer.forward(upInput);
var upIn = document.getElementById("upIn").getContext("2d");
var upOut = document.getElementById("upOut").getContext("2d");
drawFromArr(upInput, upIn, 2, 2, 3);
drawFromArr(upLayer.outData, upOut, 4, 4, 3);

//MAX POOL CODE
var mpLayer = new MaxPoolingLayer([6, 6, 3], [2, 2], 2);
// prettier-ignore
var mpLayerInput = [
	//red pixel values
	1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 
	1, 0.6, 0, 0, 0, 0, 
	0, 0.1, 0, 0, 0, 0,

	//green pixel values
	0.5, 0, 1, 0, 0, 0, 
	0.9, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 
	0.3, 0, 1, 0, 0, 0, 
	0, 0.7, 0, 0, 0, 0,

	//blue pixel values
	0, 0, 0, 0, 1, 0, 
	0, 0.9, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 
	0, 0.2, 0, 0, 1, 0, 
	0.1, 0.5, 0, 0, 0, 0
];
mpLayer.forward(mpLayerInput);
var mpIn = document.getElementById("mpIn").getContext("2d");
var mpOut = document.getElementById("mpOut").getContext("2d");
drawFromArr(mpLayerInput, mpIn, 6, 6, 3);
drawFromArr(mpLayer.outData, mpOut, 3, 3, 3);

//AVERAGE POOL LAYER CODE

var avgLayer = new AvgPoolingLayer([6, 6, 3], [2, 2], 2);
// prettier-ignore
var avgLayerInput = [
	//red pixel values
	1, 1, 0, 0, 1, 0, 
	1, 1, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 1, 
	1, 0.6, 0, 0, 0, 0, 
	0, 0.1, 0, 0, 0, 0,

	//green pixel values
	0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 1, 0, 
	0, 0, 0, 0, 1, 0, 
	0.3, 0, 1, 0, 1, 0, 
	0, 0.7, 0, 0, 1, 0,

	//blue pixel values
	1, 1, 0, 0, 0, 1, 
	1, 1, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 
	0, 0.2, 0, 0, 0, 1, 
	0.1, 0.5, 0, 0, 0, 1
];
avgLayer.forward(avgLayerInput);
var avgIn = document.getElementById("avgIn").getContext("2d");
var avgOut = document.getElementById("avgOut").getContext("2d");
drawFromArr(avgLayerInput, avgIn, 6, 6, 3);
drawFromArr(avgLayer.outData, avgOut, 3, 3, 3);
