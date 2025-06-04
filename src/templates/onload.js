document.addEventListener('DOMContentLoaded', function() {
	{{scripts[
		resolved.map(d => `(() => {\n${getData(d)}})();`).join("\n")
	]}}
});

