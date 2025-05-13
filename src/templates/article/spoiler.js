const reveal = e => console.log(e.target.closest(".spoiler")) || e.target.closest(".spoiler").classList.remove('spoiler') && e.removeEventListener(reveal);

document.querySelectorAll("div.code-wrapper.spoiler > pre").forEach(el => 
	el.addEventListener('click', reveal)
)
