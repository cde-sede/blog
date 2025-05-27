import * as fs from 'fs';
import { join, dirname, extname, relative } from 'path';

// import HtmlPrettify from 'html-prettify';
import Prettify from 'js-beautify';
import Prism from 'prismjs';
import { renderToString } from 'katex';
import { langIcons } from './icons.js';

import MarkdownIt from 'markdown-it'
import markdownItKatex from 'markdown-it-katex'

import 'prismjs/components/prism-javascript.js';
import 'prismjs/components/prism-json.js';
import 'prismjs/components/prism-python.js';
import 'prismjs/components/prism-css.js';
import 'prismjs/components/prism-c.js';
import 'prismjs/components/prism-cpp.js';
import 'prismjs/components/prism-rust.js';
import 'prismjs/components/prism-makefile.js';
import 'prismjs/components/prism-bash.js';


Prism.hooks.add('before-tokenize', function(env) {
  if (!env.grammar.plain) {
    env.grammar.plain = /\S+/;
  }
});

function prismMarkdownItPlugin(md) {
	const defaultFence = md.renderer.rules.fence || function(tokens, idx, options, env, slf) {
		return slf.renderToken(tokens, idx, options);
	};

	md.renderer.rules.fence = (tokens, idx, options, env, self) => {
		const token = tokens[idx];
		const info = token.info.trim();
		const lang = info.split(/\s+/g)[0] || 'default';

		if (!Prism.languages[lang]) {
			try {
				loadLanguages([lang]);
			} catch (e) {
				// fallback to plain if Prism doesn't know the lang
			}
		}

		// Split lines and parse custom metadata like:
		// # filename: main.js
		// # maxheight: 20
		const rawLines = token.content.split('\n');
		const params = {
			filename: lang,
			class: '',
			style: '',
		};

		while (true) {
			const line = rawLines[0];
			const match = /^#\s+(\w+):\s+(.+)$/.exec(line);
			if (!match) break;
			params[match[1]] = match[2];
			rawLines.shift();
		}

		if (params.nofile) params.filename = lang;
		if (params.maxheight) params.style += `max-height: ${params.maxheight}em;`;
		const spoilerClass = params.spoiler === 'true' ? ' spoiler' : '';
		const finalClass = (params.class || '') + spoilerClass;

		const code = rawLines.join('\n');
		const highlighted = Prism.highlight(code, Prism.languages[lang] || Prism.languages.plain, lang);

		return `
		<div class="code-wrapper${spoilerClass}">
			<div class="code-header">
				<div>
					<img class="code-icon" src="icons/${langIcons[lang] || langIcons.default}" alt="${lang} icon" />
					<span class="code-filename">${params.filename}</span>
				</div>
				<button class="copy-button" type="button" title="Copy code"><i class="fas fa-copy"></i></button>
			</div>

			<pre class="${finalClass}" style="${params.style}"><code class="language-${lang}">${highlighted}</code></pre>
		</div>`;
	};
}

function wrapWithMarkdownDiv(md) {
	const originalRender = md.render;
	md.render = function (...args) {
		const html = originalRender.apply(this, args);
		return `<div class="markdown">${html}</div>`;
	};
}

const mdConverter = new MarkdownIt({ html: true, highlight: () => '' })
	.use(prismMarkdownItPlugin)
	.use(wrapWithMarkdownDiv)
	.use(markdownItKatex, { throwOnError: false, trust: true, strict: false});


// TODO caching
const args = process.argv.slice(1);
const eIndex = args.indexOf('-e');
const eval_tag = eIndex !== -1 && args[eIndex + 1] ? args[eIndex + 1] : null;

const schema = JSON.parse(fs.readFileSync(join("src", "schema.json")));
const templates = join("src", "templates");
const output_path = join("dist")

const ensureDir = dir => !fs.existsSync(dirname(dir)) && fs.mkdirSync(dirname(dir), { recursive: true }) || true;
const getPrettifier = l => 
	l === 'html' ? s => Prettify.html(s, { indent_size: 4 }) :
	l === 'js' ? s => Prettify.js(s, { indent_size: 4, space_in_empty_paren: true }) :
	l === 'css' ? s => Prettify.css(s, { indent_size: 4 }) :
	l === 'md' ? s => mdConverter.render(s) :
	l === 'tex' ? s => renderToString(s, { displayMode: true, throwOnError: false, trust: true, strict: false}) :
	l === 'itex' ? s => renderToString(s, { displayMode: false, throwOnError: false, trust: true, strict: false}) :
			s => s;

const iprettify = (l, s) => getPrettifier(l)(s);
const prettify = (p, s) => iprettify(extname(p)?.slice?.(1), s);
const save = (p, s) => p !== "null" && ensureDir(p) && fs.writeFileSync(p, prettify(p, s), 'utf-8');
const copy = (s, d) => listFiles(s).forEach(p => ensureDir(p) && !fs.lstatSync(p).isDirectory() && fs.cpSync(p, join(d, relative(s, p))));
const listFiles = p => fs.readdirSync(p).map(e => fs.lstatSync(join(p, e)).isDirectory() ? listFiles(join(p, e)) : join(p, e)).flat();

export const Text = {
	NL: "\n",
}


mdConverter.renderer.rules.math_inline = (tokens, idx) =>
	iprettify("itex", tokens[idx].content);

mdConverter.renderer.rules.math_block = (tokens, idx) =>
	iprettify("tex", tokens[idx].content);
//	'<p>' + katex.renderToString(tokens[idx].content, {
//		throwOnError: false,
//		output: 'mathml',
//		displayMode: true
//	}) + '</p>';

const getData = (data, env, name) => {
	if (typeof data === "object") {
		if (data.type === 'isolatedobject')
			return fromObject({...data, [name]: undefined});
		if (data.type === 'object')
			return fromObject({...env, ...data, [name]: undefined});
		if (data.type === 'link')
			return fromObject({...data, ...schema[getData(data.value)]});
		if (data.type === 'file')
			return fromFile(getData(data.value), data);
		if (data.type === 'olink')
			return {...data, ...schema[data.value]};
		if (data.type === 'copy') {
			copy(join(templates, data.src), join(output_path, data.dst));
			return listFiles(join(templates, data.src));
		}
		return data
	} else return data;
};

const build = (s, data) => {
	for (const key in data) {
		const entry = data[key];
		const placeholder = `{{${key}(\\[[^]*?\\])?}}`;
		const resolved = getData(entry, data, key);
		s = s.replace(new RegExp(placeholder, 'gm'), (_, c) =>
			c ? eval(c.slice(1, -1)) : data.code ? eval(data.code) : resolved);
	}
	s = s.replace(/{{execute\[([^]*?)\]}}/, (_, c) => eval(c));
	if (data.output && data.output !== "null")
		save(join("dist", data.output !== "null" ? data.output || data.filename : "null"), s, 'utf-8');
	return s;
};

const fromObject = obj => build(obj.filename ? fs.readFileSync(join(templates, obj.filename)).toString() : obj.template || "", obj);
const fromFile = (path, data) => build(fs.readFileSync(join(templates, path)).toString(), data || {}).toString();

if (eval_tag) {
	console.log("Evaluating", eval_tag)
	console.log(fromObject(schema[eval_tag]))
} else {
	ensureDir(output_path)
	schema.pages.forEach(page => fromObject(schema[page]))
	console.log("Build done")
}
