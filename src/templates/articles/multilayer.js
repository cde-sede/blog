class NeuralNetwork {
	constructor() {
		this.w1 = Array(2).fill().map(() => Array(2).fill().map(() => Math.random() - 0.5));
		this.w2 = Array(2).fill().map(() => Math.random() - 0.5);
		this.b1 = Array(2).fill(0);
		this.b2 = 0;
		this.learningRate = 3;
	}

	sigmoid(x) {
		return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
	}

	sigmoidDerivative(x) {
		return x * (1 - x);
	}

	forward(input) {
		this.z1 = Array(2);
		this.a1 = Array(2);
		for (let i = 0; i < 2; i++) {
			this.z1[i] = this.w1[i][0] * input[0] + this.w1[i][1] * input[1] + this.b1[i];
			this.a1[i] = this.sigmoid(this.z1[i]);
		}
		this.z2 = this.w2[0] * this.a1[0] + this.w2[1] * this.a1[1] + this.b2;
		this.a2 = this.sigmoid(this.z2);
		return this.a2;
	}

	train(input, target) {
		const output = this.forward(input);
		const error = target - output;
		const delta2 = error * this.sigmoidDerivative(output);
		const delta1 = Array(2);
		for (let i = 0; i < 2; i++)
			delta1[i] = delta2 * this.w2[i] * this.sigmoidDerivative(this.a1[i]);
		for (let i = 0; i < 2; i++)
			this.w2[i] += this.learningRate * delta2 * this.a1[i];
		this.b2 += this.learningRate * delta2;
		for (let i = 0; i < 2; i++) {
			for (let j = 0; j < 2; j++)
				this.w1[i][j] += this.learningRate * delta1[i] * input[j];
			this.b1[i] += this.learningRate * delta1[i];
		}
	}

	predict(input) {
		return this.forward(input);
	}
}

let w = {
	network: new NeuralNetwork(),
	ctx: canvas.getContext('2d'),
	trainingData: [
		{ input: [0, 0], output: 0 },
		{ input: [0, 1], output: 1 },
		{ input: [1, 0], output: 1 },
		{ input: [1, 1], output: 0 }
	],
	hexToRgb: hex => {
		const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
		return result ? {
			r: parseInt(result[1], 16),
			g: parseInt(result[2], 16),
			b: parseInt(result[3], 16)
		} : {r: 0, g: 0, b: 0};
	},
	getColor: name => w.hexToRgb(getComputedStyle(document.documentElement).getPropertyValue(name).trim()),
	lerp: (s, e, t) => (e - s) * t + s,
	lerpColor: (color1, color2, t) => ({
		r: Math.round(w.lerp(color1.r, color2.r, t)),
		g: Math.round(w.lerp(color1.g, color2.g, t)),
		b: Math.round(w.lerp(color1.b, color2.b, t))
	}),
	epochsPerFrame: 50,
	maxEpochs: 5000,
	reportInterval: 50,
	epoch: 0,
	trainStep: () => {
		for (let i = 0; i < w.epochsPerFrame && w.epoch < w.maxEpochs; i++) {
			for (let example of w.trainingData)
			w.network.train(example.input, example.output);
			w.epoch++;
		}

		w.drawDecisionBoundary();
		w.populateprogress();

		if (w.epoch >= w.maxEpochs) {
			w.populateresult();
			return;
		}

		if (w.epoch < w.maxEpochs)
			requestAnimationFrame(w.trainStep);
	},
	drawDecisionBoundary: () => {

		const imageData = w.ctx.createImageData(canvas.width, canvas.height);
		const data = imageData.data;
		w.ctx.clearRect(0, 0, canvas.width, canvas.height);

		let primaryRgb = w.getColor("--accent-color");
		let secondaryRgb = w.getColor("--code-bg-color");

		for (let x = 0; x < canvas.width; x++) {
			for (let y = 0; y < canvas.height; y++) {
				const inputX = x / canvas.width;
				const inputY = 1 - (y / canvas.height);
				const prediction = w.network.predict([inputX, inputY]);

				const t = 1 / (1 + Math.exp(-12 * prediction + 6));
				const color = w.lerpColor(primaryRgb, secondaryRgb, t);
				const pixelIndex = (y * canvas.width + x) * 4;

				data[pixelIndex] = color.r;
				data[pixelIndex + 1] = color.g;
				data[pixelIndex + 2] = color.b;
				data[pixelIndex + 3] = 255;
			}
		}
		w.ctx.putImageData(imageData, 0, 0);
	},
	getError: () => {
		let totalError = 0;
		for (let example of w.trainingData) {
			const prediction = w.network.predict(example.input);
			totalError += Math.abs(example.output - prediction);
		}
		return totalError / 4
	},
	populateprogress: () => {
		progress.textContent = `Epoch ${w.epoch} / ${w.maxEpochs} Error: ${w.getError().toFixed(4)}`;
		result00.textContent = "";
		result01.textContent = "";
		result10.textContent = "";
		result11.textContent = "";
	},
	populateresult: () => {
		progress.textContent = "";
		result00.textContent = `[0, 0] -> ${w.network.predict([0, 0]).toFixed(4)}`;
		result01.textContent = `[0, 1] -> ${w.network.predict([0, 1]).toFixed(4)}`;
		result10.textContent = `[1, 0] -> ${w.network.predict([1, 0]).toFixed(4)}`;
		result11.textContent = `[1, 1] -> ${w.network.predict([1, 1]).toFixed(4)}`;
	},
	trainAndVisualize: () => {
		w.resetNetwork();
		requestAnimationFrame(w.trainStep);
	},
	resetNetwork: () => {
		w.epoch = 0;
		w.network = new NeuralNetwork();
		w.ctx.clearRect(0, 0, canvas.width, canvas.height);
		w.drawDecisionBoundary();
	},
};

w.resetNetwork();
window.w = w;
