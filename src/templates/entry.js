document.documentElement.setAttribute("data-theme", localStorage.getItem('theme') || "light")
document.addEventListener('DOMContentLoaded', function() {
	{{scripts[
		resolved.map(d => `(() => {\n${getData(d)}})();`).join("\n")
	]}}
});

