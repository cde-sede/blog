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

document.addEventListener('click', function (e) {
	const btn = e.target.closest('.copy-button');
	if (!btn) return;

	const wrapper = btn.closest('.code-wrapper');
	const code = wrapper?.querySelector('pre code');
	if (!code) return;

	const text = code.innerText;

	navigator.clipboard.writeText(text).then(() => {
		// Change icon to check
		btn.innerHTML = '<i class="fas fa-check"></i>';

		// Add "Copied" popup
		let popup = wrapper.querySelector('.copy-popup');
		if (!popup) {
			popup = document.createElement('div');
			popup.className = 'copy-popup';
			popup.textContent = 'Code copied';
			wrapper.querySelector('.code-header').appendChild(popup);
		}

		popup.classList.add('show');

		setTimeout(() => {
			btn.innerHTML = '<i class="fas fa-copy"></i>';
			popup.classList.remove('show');
		}, 1500);
	});
});
