

utils = {
	lerp: (min, max, x) => x * (max - min) + min,
	easeOutQuad: x => 1 - (1 - x) * (1 - x),
	animate: (a, b, duration, ease, render) => new Promise(resolve => {
		let start;
		const step = now => {
			start ??= now;
			const fraction = Math.min(1, (now - start) / duration);
			render(utils.lerp(a, b, ease(fraction)));
			if (fraction < 1) requestAnimationFrame(step)
				else resolve();
		}
		requestAnimationFrame(step);
	}),
};

window.utils = utils;
