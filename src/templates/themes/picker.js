
const picker = {
	el: document.getElementById("picker"),
	shown: false,
	show: () => {
		picker.el.classList.add("shown"),
		picker.shown = true;
	},
	hide: () => {
		picker.el.classList.remove("shown");
		picker.shown = false;
	},
	run: () => {
		document.getElementById("picker-button").addEventListener("click", () => 
			picker.shown ? picker.hide() : picker.show()
		);
		window.addEventListener("click", e => {
			if (!(e.target.closest("#picker") || e.target.closest("#picker-button")))
				picker.hide()
		});

		document.querySelectorAll('.theme-option').forEach(el => {
			const bgcolor = el.dataset.bgcolor;
			const textcolor = el.dataset.textcolor;
			const accentcolor = el.dataset.accentcolor;

			let anim = null;
			const start_percent = 20;
			const end_percent = 65;
			let currentPercent = start_percent;
			let lastFramePercent = start_percent;


			// TODO Sheath effect on an element in front of the text
			const setGradient = percent => 
				el.style.background = `linear-gradient(135deg, ${bgcolor} 0%, ${bgcolor} ${percent}%, ${accentcolor} ${percent}%, ${accentcolor} ${percent + 20}%, ${textcolor} ${percent + 20}%, ${textcolor} 100%)`;

			function animateGradient(to, duration = 300) {
				if (anim) cancelAnimationFrame(anim);
				const from = lastFramePercent;
				const start = performance.now();

				function frame(now) {
					const t = Math.min(1, (now - start) / duration);
					const val = from + (to - from) * t;
					lastFramePercent = val;
					setGradient(val);
					if (t < 1) {
						anim = requestAnimationFrame(frame);
					} else {
						currentPercent = to;
					}
				}

				anim = requestAnimationFrame(frame);
			}

			el.addEventListener("mouseenter", () => animateGradient(end_percent));
			el.addEventListener("mouseleave", () => animateGradient(start_percent));

			setGradient(currentPercent);
		})



	},
	theme: theme => {
		console.log(theme);
		document.documentElement.setAttribute("data-theme", theme);
		localStorage.setItem("theme", theme);
	},
}

window.picker = picker;
picker.run()



