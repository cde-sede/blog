const start_percent = 20;
const end_percent = 65;

const picker = {
	el: document.getElementById("picker"),
	selected: localStorage.getItem('theme'),
	shown: false,
	show: () => {
		picker.el.classList.add("shown");
		picker.shown = true;
	},
	hide: () => {
		picker.el.classList.remove("shown");
		picker.shown = false;
	},
	themes: {},
	run: () => {
		document.getElementById("picker-button").addEventListener("click", () => 
			picker.shown ? picker.hide() : picker.show()
		);
		window.addEventListener("click", e => {
			if (!(e.target.closest("#picker") || e.target.closest("#picker-button")))
				picker.hide();
		});
		
		picker.el.querySelectorAll('.theme-option').forEach(el => {
			const bgcolor = el.dataset.bgcolor;
			const textcolor = el.dataset.textcolor;
			const accentcolor = el.dataset.accentcolor;
			let anim = null;
			let currentPercent = start_percent;
			const themeName = el.dataset.theme;
			
			if (picker.selected === themeName) {
				currentPercent = end_percent;
			}
			
			const setGradient = percent => 
				el.style.background = `linear-gradient(135deg, ${bgcolor} 0%, ${bgcolor} ${percent}%, ${accentcolor} ${percent}%, ${accentcolor} ${percent + 20}%, ${textcolor} ${percent + 20}%, ${textcolor} 100%)`;
			
			function animateGradient(to, duration = 300) {
				if (anim) {
					anim.cancel();
					anim = null;
				}
				
				let active = true;
				const cancelFn = { cancel: () => { active = false; } };
				const from = currentPercent;
				
				window.utils.animate(
					from, to, duration, t => t, (val) => {
						if (active) {
							currentPercent = val;
							setGradient(val);
						}
					}
				).then(() => {
					if (active) {
						currentPercent = to;
					}
				});
				
				anim = cancelFn;
				return cancelFn;
			}
			
			const handleMouseEnter = () => {
				if (picker.selected !== themeName) {
					animateGradient(end_percent);
				}
			};
			
			const handleMouseLeave = () => {
				if (picker.selected !== themeName) {
					animateGradient(start_percent);
				}
			};
			
			el.addEventListener("mouseenter", handleMouseEnter);
			el.addEventListener("mouseleave", handleMouseLeave);
			
			picker.themes[themeName] = {
				el,
				isSelected: false,
				currentPercent,
				animateGradient,
				setGradient,
				handleMouseEnter,
				handleMouseLeave
			};
			
			setGradient(currentPercent);
			
			el.addEventListener("click", () => {
				picker.theme(themeName);
			});
		});
		
		if (picker.selected && picker.themes[picker.selected]) {
			picker.themes[picker.selected].isSelected = true;
		}
	},
	theme: theme => {
		if (picker.selected === theme) return;
		
		if (picker.selected && picker.themes[picker.selected]) {
			const prevTheme = picker.themes[picker.selected];
			prevTheme.isSelected = false;
			prevTheme.animateGradient(start_percent);
		}
		
		const newTheme = picker.themes[theme];
		newTheme.isSelected = true;
		newTheme.animateGradient(end_percent);

		picker.selected = theme;
		document.documentElement.setAttribute("data-theme", theme);
		localStorage.setItem("theme", theme);
	},
}

window.picker = picker;
picker.run()




