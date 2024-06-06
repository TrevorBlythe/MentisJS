function generateSierpinskiTriangle(width, height, depth, scale) {
	// Initialize empty image data array
	const imageData = new Array(width * height).fill(0);

	// Define initial triangle coordinates (can be learned)
	const point1 = { x: width / 2, y: height };
	const point2 = { x: 0, y: 0 };
	const point3 = { x: width, y: 0 };

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			// Randomly choose a point within the initial triangle
			let currentPoint = chooseRandomPoint(point1, point2, point3);

			// Iterate for a certain depth (can be learned)
			for (let i = 0; i < depth; i++) {
				// Choose a random corner of the triangle
				const randomCorner = chooseRandomCorner(point1, point2, point3);

				// Calculate midpoint between current point and chosen corner
				currentPoint = {
					x: (currentPoint.x + randomCorner.x) / 2,
					y: (currentPoint.y + randomCorner.y) / 2,
				};
			}

			// If the point falls within the final triangle, mark the pixel as 'on' (value of 1)
			if (isPointInsideTriangle(currentPoint, point1, point2, point3)) {
				imageData[y * width + x] = 1;
			}
		}
	}

	return imageData;
}

// Helper functions to calculate points and triangle area
function chooseRandomPoint(point1, point2, point3) {
	const r1 = Math.random();
	const r2 = Math.random();
	return {
		x: point1.x + (point2.x - point1.x) * r1 + (point3.x - point1.x - point2.x + point1.x) * r1 * r2,
		y: point1.y + (point2.y - point1.y) * r1 + (point3.y - point1.y - point2.y + point1.y) * r1 * r2,
	};
}

function chooseRandomCorner(point1, point2, point3) {
	const randomValue = Math.random();
	if (randomValue < 1 / 3) {
		return point1;
	} else if (randomValue < 2 / 3) {
		return point2;
	} else {
		return point3;
	}
}

function isPointInsideTriangle(point, point1, point2, point3) {
	const area = Math.abs(
		(point1.x * (point2.y - point3.y) + point2.x * (point3.y - point1.y) + point3.x * (point1.y - point2.y)) / 2
	);
	const area1 = Math.abs(
		(point.x * (point2.y - point3.y) + point2.x * (point3.y - point.y) + point3.x * (point.y - point2.y)) / 2
	);
	const area2 = Math.abs(
		(point1.x * (point.y - point3.y) + point.x * (point3.y - point1.y) + point3.x * (point1.y - point.y)) / 2
	);
	const area3 = Math.abs(
		(point1.x * (point2.y - point.y) + point2.x * (point.y - point1.y) + point.x * (point1.y - point2.y)) / 2
	);
	return area1 + area2 + area3 === area;
}

function generateMandelbrotSet(width, height, maxIterations) {
	const imageData = new Array(width * height * 4).fill(0); // RGBA format

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const real = (x / (width - 1)) * 3.5 - 2.5; // Scale coordinates to Mandelbrot set range
			const imaginary = (y / (height - 1)) * 2.0;

			let a = 0.0,
				b = 0.0,
				iteration = 0;

			while (a * a + b * b <= 4.0 && iteration < maxIterations) {
				const tempA = a * a - b * b + real;
				b = 2.0 * a * b + imaginary;
				a = tempA;
				iteration++;
			}

			// Color based on number of iterations (escape time)
			let color;
			if (iteration === maxIterations) {
				color = [0, 0, 0, 255]; // Black (maximum iterations reached)
			} else {
				const colorScale = iteration / maxIterations;
				color = [
					Math.floor(colorScale * 255), // Red
					Math.floor(colorScale * 128), // Green
					Math.floor(colorScale * 64), // Blue
					255, // Alpha (always opaque)
				];
			}

			imageData.splice(y * width * 4 + x * 4, 0, ...color); // Set RGBA values
		}
	}

	return imageData;
}

function generateWeierstrassFern(width, height, numIterations, scale) {
	const imageData = new Array(width * height).fill(0);

	// Define Barnsley Fern parameters (can be learned)
	const fernProbabilities = [0.85, 0.1, 0.04, 0.01];
	const fernTransforms = [
		// Scale down, move left
		(x, y) => ({ x: 0.85 * x, y: 0.04 * y - 0.1 }),
		// Scale down, move right
		(x, y) => ({ x: 0.15 * x, y: 0.85 * y + 1.3 }),
		// Scale down and reflect over x-axis, move left
		(x, y) => ({ x: 0.2 * x, y: -0.25 * y + 0.2 }),
		// Reflect over x and y-axis, move right
		(x, y) => ({ x: 0.2 * x + 0.23, y: -0.25 * y - 1.6 }),
	];

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			let currentPoint = { x: 0, y: 0 }; // Starting point

			// Iterate using Barnsley Fern steps
			for (let i = 0; i < numIterations; i++) {
				const randomValue = Math.random();
				let chosenTransformIndex = 0;
				let cumulativeSum = fernProbabilities[0];
				while (cumulativeSum < randomValue && chosenTransformIndex < fernTransforms.length - 1) {
					chosenTransformIndex++;
					cumulativeSum += fernProbabilities[chosenTransformIndex];
				}
				currentPoint = fernTransforms[chosenTransformIndex](currentPoint.x, currentPoint.y);
			}

			// Apply Weierstrass function (simplified version)
			let weierstrassValue = 0;
			const frequency = 5;
			const amplitude = 0.1;
			for (let n = 1; n < 10; n++) {
				// Adjust loop iterations for complexity
				const term = Math.sin(n * frequency * currentPoint.x);
				const power = Math.pow(0.5, n);
				weierstrassValue += amplitude * term * power;
			}

			// Normalize and convert to grayscale value (0-255)
			const grayscale = Math.floor((255 * (weierstrassValue + 1)) / 2); // Map to 0-255 range

			imageData[y * width + x] = grayscale;
		}
	}

	return imageData;
}

function generateJuliaSet(width, height, maxIterations, c) {
	const imageData = new Array(width * height).fill(0);

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			// Normalize coordinates to complex plane
			const real = (x / (width - 1)) * 3.0 - 1.5;
			const imaginary = (y / (height - 1)) * 2.0;

			let a = real,
				b = imaginary,
				iteration = 0;

			// Iterate based on Julia set definition
			while (a * a + b * b <= 4.0 && iteration < maxIterations) {
				const tempA = a * a - b * b + c.real;
				b = 2.0 * a * b + c.imag;
				a = tempA;
				iteration++;
			}

			// Color based on number of iterations (escape time)
			let color;
			if (iteration === maxIterations) {
				color = 0; // Black (maximum iterations reached)
			} else {
				const colorScale = iteration / maxIterations;
				color = Math.floor(colorScale * 255); // Grayscale based on escape time
			}

			imageData[y * width + x] = color;
		}
	}

	return imageData;
}
function generateSimplifiedMandelbrot(width, height, maxIterations, offsetX, offsetY, scaleX = 1, scaleY = 1) {
	const imageData = new Array(width * height).fill(0);

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			// Normalize coordinates with offset and scaling
			const real = (x / (width / 2 - 1)) * scaleX + offsetX;
			const imaginary = (y / (height / 2 - 1)) * scaleY + offsetY;

			let a = real,
				b = imaginary,
				iteration = 0;
			const threshold = 4.0; // Early termination threshold

			// Simplified iteration with early termination
			while (a * a + b * b <= threshold && iteration < maxIterations) {
				const tempA = a * a - b * b + real;
				b = 2.0 * a * b + imaginary;
				a = tempA;
				iteration++;
			}

			// Assign value based on number of iterations (escape time)
			imageData[y * width + x] = iteration === maxIterations ? 0 : iteration * 255; // 0 for maxIterations, iteration count otherwise
		}
	}

	return imageData;
}
