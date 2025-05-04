const excerpts = document.querySelectorAll(".excerpt-wrapper");
if (!excerpts)
	return ;

excerpts.forEach((el, i) => el.setAttribute('data-index', i))

const observer = new IntersectionObserver(entries => {
	entries.forEach(entry => {
		if (entry.isIntersecting) {
			entry.target.style.transitionDelay = `${entry.target.dataset.index * 100}ms`;
			entry.target.classList.add('visible');
			observer.unobserve(entry.target);
		}
	});
}, { threshold: 0.1 });

const style = document.createElement('style');
document.head.appendChild(style);

excerpts.forEach(el => {
	observer.observe(el);
});
