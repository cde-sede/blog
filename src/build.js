import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'fs';
import { join, dirname, extname } from 'path';

// import HtmlPrettify from 'html-prettify';
import Prettify from 'js-beautify';
import Showdown from 'showdown';
import Prism from "prismjs";
import { load } from 'cheerio';

import 'prismjs/components/prism-javascript.js';
import 'prismjs/components/prism-python.js'; // example

const langIcons = {
	aarch64:                 "aarch64-original.svg",
	adonisjs:                "adonisjs-original.svg",
	aftereffects:            "aftereffects-original.svg",
	akka:                    "akka-original.svg",
	algolia:                 "algolia-original.svg",
	alpinejs:                "alpinejs-original.svg",
	anaconda:                "anaconda-original.svg",
	android:                 "android-original.svg",
	androidstudio:           "androidstudio-original.svg",
	angular:                 "angular-original.svg",
	angularjs:               "angularjs-original.svg",
	angularmaterial:         "angularmaterial-original.svg",
	ansible:                 "ansible-original.svg",
	antdesign:               "antdesign-original.svg",
	apache:                  "apache-original.svg",
	apacheairflow:           "apacheairflow-original.svg",
	apachekafka:             "apachekafka-original.svg",
	apachespark:             "apachespark-original.svg",
	apl:                     "apl-original.svg",
	appcelerator:            "appcelerator-original.svg",
	apple:                   "apple-original.svg",
	appwrite:                "appwrite-original.svg",
	archlinux:               "archlinux-original.svg",
	arduino:                 "arduino-original.svg",
	argocd:                  "argocd-original.svg",
	astro:                   "astro-original.svg",
	atom:                    "atom-original.svg",
	azure:                   "azure-original.svg",
	azuredevops:             "azuredevops-original.svg",
	azuresqldatabase:        "azuresqldatabase-original.svg",
	babel:                   "babel-original.svg",
	backbonejs:              "backbonejs-original.svg",
	ballerina:               "ballerina-original.svg",
	bamboo:                  "bamboo-original.svg",
	bash:                    "bash-original.svg",
	beats:                   "beats-original.svg",
	behance:                 "behance-original.svg",
	bitbucket:               "bitbucket-original.svg",
	blazor:                  "blazor-original.svg",
	blender:                 "blender-original.svg",
	bootstrap:               "bootstrap-original.svg",
	bower:                   "bower-original.svg",
	browserstack:            "browserstack-original.svg",
	bun:                     "bun-original.svg",
	c:                       "c-original.svg",
	cairo:                   "cairo-original.svg",
	cakephp:                 "cakephp-original.svg",
	canva:                   "canva-original.svg",
	capacitor:               "capacitor-original.svg",
	carbon:                  "carbon-original.svg",
	cassandra:               "cassandra-original.svg",
	centos:                  "centos-original.svg",
	ceylon:                  "ceylon-original.svg",
	chrome:                  "chrome-original.svg",
	clarity:                 "clarity-original.svg",
	clion:                   "clion-original.svg",
	clojure:                 "clojure-original.svg",
	clojurescript:           "clojurescript-original.svg",
	cloudflare:              "cloudflare-original.svg",
	cloudflareworkers:       "cloudflareworkers-original.svg",
	cmake:                   "cmake-original.svg",
	codeac:                  "codeac-original.svg",
	codepen:                 "codepen-original.svg",
	coffeescript:            "coffeescript-original.svg",
	composer:                "composer-original.svg",
	confluence:              "confluence-original.svg",
	consul:                  "consul-original.svg",
	contao:                  "contao-original.svg",
	corejs:                  "corejs-original.svg",
	cosmosdb:                "cosmosdb-original.svg",
	couchbase:               "couchbase-original.svg",
	couchdb:                 "couchdb-original.svg",
	cplusplus:               "cplusplus-original.svg",
	crystal:                 "crystal-original.svg",
	csharp:                  "csharp-original.svg",
	css3:                    "css3-original.svg",
	cypressio:               "cypressio-original.svg",
	d3js:                    "d3js-original.svg",
	dart:                    "dart-original.svg",
	datagrip:                "datagrip-original.svg",
	dataspell:               "dataspell-original.svg",
	dbeaver:                 "dbeaver-original.svg",
	debian:                  "debian-original.svg",
	denojs:                  "denojs-original.svg",
	devicon:                 "devicon-original.svg",
	digitalocean:            "digitalocean-original.svg",
	discordjs:               "discordjs-original.svg",
	djangorest:              "djangorest-original.svg",
	docker:                  "docker-original.svg",
	doctrine:                "doctrine-original.svg",
	dot:                     "dot-net-original.svg",
	dotnetcore:              "dotnetcore-original.svg",
	dreamweaver:             "dreamweaver-original.svg",
	dropwizard:              "dropwizard-original.svg",
	drupal:                  "drupal-original.svg",
	dynamodb:                "dynamodb-original.svg",
	eclipse:                 "eclipse-original.svg",
	ecto:                    "ecto-original.svg",
	elasticsearch:           "elasticsearch-original.svg",
	electron:                "electron-original.svg",
	eleventy:                "eleventy-original.svg",
	elixir:                  "elixir-original.svg",
	elm:                     "elm-original.svg",
	emacs:                   "emacs-original.svg",
	email:                   "email.svg",
	embeddedc:               "embeddedc-original.svg",
	ember:                   "ember-original.svg",
	envoy:                   "envoy-original.svg",
	erlang:                  "erlang-original.svg",
	eslint:                  "eslint-original.svg",
	express:                 "express-original.svg",
	facebook:                "facebook-original.svg",
	fastapi:                 "fastapi-original.svg",
	fastify:                 "fastify-original.svg",
	faunadb:                 "faunadb-original.svg",
	feathersjs:              "feathersjs-original.svg",
	fedora:                  "fedora-original.svg",
	figma:                   "figma-original.svg",
	filezilla:               "filezilla-original.svg",
	firebase:                "firebase-original.svg",
	firefox:                 "firefox-original.svg",
	flask:                   "flask-original.svg",
	flutter:                 "flutter-original.svg",
	fortran:                 "fortran-original.svg",
	foundation:              "foundation-original.svg",
	framermotion:            "framermotion-original.svg",
	framework7:              "framework7-original.svg",
	fsharp:                  "fsharp-original.svg",
	gatling:                 "gatling-original.svg",
	gatsby:                  "gatsby-original.svg",
	gazebo:                  "gazebo-original.svg",
	gcc:                     "gcc-original.svg",
	gentoo:                  "gentoo-original.svg",
	ghost:                   "ghost-original.svg",
	gimp:                    "gimp-original.svg",
	git:                     "git-original.svg",
	gitbook:                 "gitbook-original.svg",
	github:                  "github-original.svg",
	github:                  "github.svg",
	githubactions:           "githubactions-original.svg",
	githubcodespaces:        "githubcodespaces-original.svg",
	gitlab:                  "gitlab-original.svg",
	gitpod:                  "gitpod-original.svg",
	go:                      "go-original.svg",
	godot:                   "godot-original.svg",
	goland:                  "goland-original.svg",
	google:                  "google-original.svg",
	googlecloud:             "googlecloud-original.svg",
	gradle:                  "gradle-original.svg",
	grafana:                 "grafana-original.svg",
	grails:                  "grails-original.svg",
	groovy:                  "groovy-original.svg",
	grpc:                    "grpc-original.svg",
	grunt:                   "grunt-original.svg",
	hadoop:                  "hadoop-original.svg",
	handlebars:              "handlebars-original.svg",
	hardhat:                 "hardhat-original.svg",
	harvester:               "harvester-original.svg",
	haskell:                 "haskell-original.svg",
	haxe:                    "haxe-original.svg",
	helm:                    "helm-original.svg",
	heroku:                  "heroku-original.svg",
	hibernate:               "hibernate-original.svg",
	homebrew:                "homebrew-original.svg",
	html5:                   "html5-original.svg",
	hugo:                    "hugo-original.svg",
	ie10:                    "ie10-original.svg",
	ifttt:                   "ifttt-original.svg",
	influxdb:                "influxdb-original.svg",
	inkscape:                "inkscape-original.svg",
	insomnia:                "insomnia-original.svg",
	intellij:                "intellij-original.svg",
	ionic:                   "ionic-original.svg",
	jaegertracing:           "jaegertracing-original.svg",
	jamstack:                "jamstack-original.svg",
	jasmine:                 "jasmine-original.svg",
	java:                    "java-original.svg",
	javascript:              "javascript-original.svg",
	jeet:                    "jeet-original.svg",
	jekyll:                  "jekyll-original.svg",
	jenkins:                 "jenkins-original.svg",
	jetbrains:               "jetbrains-original.svg",
	jetpackcompose:          "jetpackcompose-original.svg",
	jira:                    "jira-original.svg",
	jiraalign:               "jiraalign-original.svg",
	jquery:                  "jquery-original.svg",
	json:                    "json-original.svg",
	jule:                    "jule-original.svg",
	julia:                   "julia-original.svg",
	junit:                   "junit-original.svg",
	jupyter:                 "jupyter-original.svg",
	k3os:                    "k3os-original.svg",
	k3s:                     "k3s-original.svg",
	k6:                      "k6-original.svg",
	kaggle:                  "kaggle-original.svg",
	karatelabs:              "karatelabs-original.svg",
	karma:                   "karma-original.svg",
	kdeneon:                 "kdeneon-original.svg",
	keras:                   "keras-original.svg",
	kibana:                  "kibana-original.svg",
	knexjs:                  "knexjs-original.svg",
	kotlin:                  "kotlin-original.svg",
	krakenjs:                "krakenjs-original.svg",
	ktor:                    "ktor-original.svg",
	kubernetes:              "kubernetes-original.svg",
	labview:                 "labview-original.svg",
	laravel:                 "laravel-original.svg",
	latex:                   "latex-original.svg",
	linkedin:                "linkedin-original.svg",
	linkedin:                "linkedin.svg",
	linux:                   "linux-original.svg",
	liquibase:               "liquibase-original.svg",
	livewire:                "livewire-original.svg",
	llvm:                    "llvm-original.svg",
	lodash:                  "lodash-original.svg",
	logstash:                "logstash-original.svg",
	lua:                     "lua-original.svg",
	lumen:                   "lumen-original.svg",
	magento:                 "magento-original.svg",
	mariadb:                 "mariadb-original.svg",
	markdown:                "markdown-original.svg",
	materializecss:          "materializecss-original.svg",
	materialui:              "materialui-original.svg",
	matlab:                  "matlab-original.svg",
	matplotlib:              "matplotlib-original.svg",
	maven:                   "maven-original.svg",
	maya:                    "maya-original.svg",
	meteor:                  "meteor-original.svg",
	microsoftsqlserver:      "microsoftsqlserver-original.svg",
	minitab:                 "minitab-original.svg",
	mithril:                 "mithril-original.svg",
	mobx:                    "mobx-original.svg",
	mocha:                   "mocha-original.svg",
	modx:                    "modx-original.svg",
	moleculer:               "moleculer-original.svg",
	mongodb:                 "mongodb-original.svg",
	mongoose:                "mongoose-original.svg",
	moodle:                  "moodle-original.svg",
	msdos:                   "msdos-original.svg",
	mysql:                   "mysql-original.svg",
	nano:                    "nano-original.svg",
	neo4j:                   "neo4j-original.svg",
	neovim:                  "neovim-original.svg",
	nestjs:                  "nestjs-original.svg",
	netlify:                 "netlify-original.svg",
	networkx:                "networkx-original.svg",
	nextjs:                  "nextjs-original.svg",
	nginx:                   "nginx-original.svg",
	ngrx:                    "ngrx-original.svg",
	nhibernate:              "nhibernate-original.svg",
	nim:                     "nim-original.svg",
	nimble:                  "nimble-original.svg",
	nixos:                   "nixos-original.svg",
	nodejs:                  "nodejs-original.svg",
	nodemon:                 "nodemon-original.svg",
	nodewebkit:              "nodewebkit-original.svg",
	nomad:                   "nomad-original.svg",
	norg:                    "norg-original.svg",
	notion:                  "notion-original.svg",
	nuget:                   "nuget-original.svg",
	numpy:                   "numpy-original.svg",
	nuxtjs:                  "nuxtjs-original.svg",
	oauth:                   "oauth-original.svg",
	ocaml:                   "ocaml-original.svg",
	ohmyzsh:                 "ohmyzsh-original.svg",
	okta:                    "okta-original.svg",
	openal:                  "openal-original.svg",
	openapi:                 "openapi-original.svg",
	opencl:                  "opencl-original.svg",
	opencv:                  "opencv-original.svg",
	opengl:                  "opengl-original.svg",
	openstack:               "openstack-original.svg",
	opensuse:                "opensuse-original.svg",
	opentelemetry:           "opentelemetry-original.svg",
	opera:                   "opera-original.svg",
	oracle:                  "oracle-original.svg",
	ory:                     "ory-original.svg",
	p5js:                    "p5js-original.svg",
	packer:                  "packer-original.svg",
	pandas:                  "pandas-original.svg",
	perl:                    "perl-original.svg",
	pfsense:                 "pfsense-original.svg",
	phalcon:                 "phalcon-original.svg",
	phoenix:                 "phoenix-original.svg",
	photonengine:            "photonengine-original.svg",
	photoshop:               "photoshop-original.svg",
	php:                     "php-original.svg",
	phpstorm:                "phpstorm-original.svg",
	picker:                  "picker.svg",
	playwright:              "playwright-original.svg",
	plotly:                  "plotly-original.svg",
	pnpm:                    "pnpm-original.svg",
	podman:                  "podman-original.svg",
	poetry:                  "poetry-original.svg",
	polygon:                 "polygon-original.svg",
	portainer:               "portainer-original.svg",
	postcss:                 "postcss-original.svg",
	postgresql:              "postgresql-original.svg",
	postman:                 "postman-original.svg",
	powershell:              "powershell-original.svg",
	premierepro:             "premierepro-original.svg",
	prisma:                  "prisma-original.svg",
	processing:              "processing-original.svg",
	prolog:                  "prolog-original.svg",
	prometheus:              "prometheus-original.svg",
	protractor:              "protractor-original.svg",
	pulsar:                  "pulsar-original.svg",
	pulumi:                  "pulumi-original.svg",
	puppeteer:               "puppeteer-original.svg",
	purescript:              "purescript-original.svg",
	putty:                   "putty-original.svg",
	pycharm:                 "pycharm-original.svg",
	pypi:                    "pypi-original.svg",
	pytest:                  "pytest-original.svg",
	python:                  "python-original.svg",
	pytorch:                 "pytorch-original.svg",
	qodana:                  "qodana-original.svg",
	qt:                      "qt-original.svg",
	quarkus:                 "quarkus-original.svg",
	quasar:                  "quasar-original.svg",
	qwik:                    "qwik-original.svg",
	r:                       "r-original.svg",
	rabbitmq:                "rabbitmq-original.svg",
	railway:                 "railway-original.svg",
	rancher:                 "rancher-original.svg",
	raspberrypi:             "raspberrypi-original.svg",
	reach:                   "reach-original.svg",
	react:                   "react-original.svg",
	reactbootstrap:          "reactbootstrap-original.svg",
	reactnavigation:         "reactnavigation-original.svg",
	reactrouter:             "reactrouter-original.svg",
	readthedocs:             "readthedocs-original.svg",
	realm:                   "realm-original.svg",
	rect:                    "rect-original.svg",
	redhat:                  "redhat-original.svg",
	redis:                   "redis-original.svg",
	redux:                   "redux-original.svg",
	renpy:                   "renpy-original.svg",
	replit:                  "replit-original.svg",
	rider:                   "rider-original.svg",
	rocksdb:                 "rocksdb-original.svg",
	rockylinux:              "rockylinux-original.svg",
	rollup:                  "rollup-original.svg",
	ros:                     "ros-original.svg",
	rspec:                   "rspec-original.svg",
	rstudio:                 "rstudio-original.svg",
	ruby:                    "ruby-original.svg",
	rubymine:                "rubymine-original.svg",
	rust:                    "rust-original.svg",
	rxjs:                    "rxjs-original.svg",
	safari:                  "safari-original.svg",
	salesforce:              "salesforce-original.svg",
	sanity:                  "sanity-original.svg",
	sass:                    "sass-original.svg",
	scala:                   "scala-original.svg",
	scalingo:                "scalingo-original.svg",
	scikitlearn:             "scikitlearn-original.svg",
	sdl:                     "sdl-original.svg",
	selenium:                "selenium-original.svg",
	sema:                    "sema-original.svg",
	sentry:                  "sentry-original.svg",
	sequelize:               "sequelize-original.svg",
	shopware:                "shopware-original.svg",
	shotgrid:                "shotgrid-original.svg",
	sketch:                  "sketch-original.svg",
	slack:                   "slack-original.svg",
	socketio:                "socketio-original.svg",
	solidity:                "solidity-original.svg",
	solidjs:                 "solidjs-original.svg",
	sonarqube:               "sonarqube-original.svg",
	sourcetree:              "sourcetree-original.svg",
	spack:                   "spack-original.svg",
	spring:                  "spring-original.svg",
	spss:                    "spss-original.svg",
	spyder:                  "spyder-original.svg",
	sqlalchemy:              "sqlalchemy-original.svg",
	sqldeveloper:            "sqldeveloper-original.svg",
	sqlite:                  "sqlite-original.svg",
	ssh:                     "ssh-original.svg",
	stackoverflow:           "stackoverflow-original.svg",
	storybook:               "storybook-original.svg",
	streamlit:               "streamlit-original.svg",
	stylus:                  "stylus-original.svg",
	subversion:              "subversion-original.svg",
	supabase:                "supabase-original.svg",
	svelte:                  "svelte-original.svg",
	swagger:                 "swagger-original.svg",
	swift:                   "swift-original.svg",
	swiper:                  "swiper-original.svg",
	symfony:                 "symfony-original.svg",
	tailwindcss:             "tailwindcss-original.svg",
	tauri:                   "tauri-original.svg",
	tensorflow:              "tensorflow-original.svg",
	terraform:               "terraform-original.svg",
	tex:                     "tex-original.svg",
	thealgorithms:           "thealgorithms-original.svg",
	threedsmax:              "threedsmax-original.svg",
	threejs:                 "threejs-original.svg",
	titaniumsdk:             "titaniumsdk-original.svg",
	tomcat:                  "tomcat-original.svg",
	tortoisegit:             "tortoisegit-original.svg",
	towergit:                "towergit-original.svg",
	traefikmesh:             "traefikmesh-original.svg",
	traefikproxy:            "traefikproxy-original.svg",
	travis:                  "travis-original.svg",
	trello:                  "trello-original.svg",
	trpc:                    "trpc-original.svg",
	twitter:                 "twitter-original.svg",
	typescript:              "typescript-original.svg",
	typo3:                   "typo3-original.svg",
	ubuntu:                  "ubuntu-original.svg",
	unifiedmodelinglanguage: "unifiedmodelinglanguage-original.svg",
	unity:                   "unity-original.svg",
	unix:                    "unix-original.svg",
	unrealengine:            "unrealengine-original.svg",
	uwsgi:                   "uwsgi-original.svg",
	v8:                      "v8-original.svg",
	vagrant:                 "vagrant-original.svg",
	vala:                    "vala-original.svg",
	vault:                   "vault-original.svg",
	vercel:                  "vercel-original.svg",
	vertx:                   "vertx-original.svg",
	vim:                     "vim-original.svg",
	visualbasic:             "visualbasic-original.svg",
	visualstudio:            "visualstudio-original.svg",
	vite:                    "vite-original.svg",
	vitejs:                  "vitejs-original.svg",
	vitess:                  "vitess-original.svg",
	vitest:                  "vitest-original.svg",
	vscode:                  "vscode-original.svg",
	vsphere:                 "vsphere-original.svg",
	vuejs:                   "vuejs-original.svg",
	vuestorefront:           "vuestorefront-original.svg",
	vuetify:                 "vuetify-original.svg",
	vyper:                   "vyper-original.svg",
	wasm:                    "wasm-original.svg",
	webflow:                 "webflow-original.svg",
	weblate:                 "weblate-original.svg",
	webpack:                 "webpack-original.svg",
	webstorm:                "webstorm-original.svg",
	windows8:                "windows8-original.svg",
	windows11:               "windows11-original.svg",
	woocommerce:             "woocommerce-original.svg",
	wordpress:               "wordpress-original.svg",
	xamarin:                 "xamarin-original.svg",
	xcode:                   "xcode-original.svg",
	xd:                      "xd-original.svg",
	xml:                     "xml-original.svg",
	yaml:                    "yaml-original.svg",
	yarn:                    "yarn-original.svg",
	yii:                     "yii-original.svg",
	yugabytedb:              "yugabytedb-original.svg",
	yunohost:                "yunohost-original.svg",
	zend:                    "zend-original.svg",
	zig:                     "zig-original.svg",

	default:                 "default.svg",
};

const prismExtension = () => [{
	type: 'output',
	filter: (html) => {
		const $ = load(html);

		$('pre code').each((_, el) => {
			const $el = $(el);
			const className = $el.attr('class') || '';
			const lang = className.match(/language-(\w+)/)?.[1] || 'default';
			const lines = $el.text().split('\n');

			let filename = null;
			if (lines[0].startsWith('# '))
				filename = lines.shift().slice(2).trim();

			const highlighted = Prism.highlight(lines.join('\n'), Prism.languages[lang] || Prism.languages.plain, lang);

			if (filename)
				$el.parent().replaceWith(`
				  <div class="code-wrapper">
					<div class="code-header">
					  <img class="code-icon" src="icons/${langIcons[lang] || langIcons.default}" alt="${lang} icon" />
					  <span class="code-filename">${filename}</span>
					</div>
					<pre class="${$el.parent().attr('class')}"><code class="${className}">${highlighted}</code></pre>
				  </div>
				`);
			else
				$el.html(highlighted);
		});

		return $.html();
	}
}];




const mdConverter = new Showdown.Converter({
	omitExtraWLInCodeBlocks: true,
	noHeaderId: true,
	customizedHeaderId: true,
	parseImgDimensions: true,
	simplifiedAutoLink: true,
	literalMidWordUnderscoresa: true,
	strikethrough: true,
	tables: true,
	tasklists: true,
	smartIndentationFix: true,
	openLinksInNewWindow: true,
	backslashEscapesHTMLTags: true,
	emoji: true,
	moreStyling: true,
	extensions: [prismExtension]
});

// TODO caching
const args = process.argv.slice(1);
const eIndex = args.indexOf('-e');
const eval_tag = eIndex !== -1 && args[eIndex + 1] ? args[eIndex + 1] : null;

let depth = 0;
let ignored = true;

const schema = JSON.parse(readFileSync(join("src", "schema.json")));
const templates = join("src", "templates");
const output_path = join("dist")

const ensureDir = dir => !existsSync(dirname(dir)) && mkdirSync(dirname(dir), { recursive: true }) || true;
const getPrettifier = l => 
	l === 'html' ? s => Prettify.html(s, { indent_size: 4 }) :
	l === 'js' ? s => Prettify.js(s, { indent_size: 4, space_in_empty_paren: true }) :
	l === 'css' ? s => Prettify.css(s, { indent_size: 4 }) :
	l === 'md' ? s => mdConverter.makeHtml(s) :
			s => s;

const iprettify = (l, s) => getPrettifier(l)(s);
const prettify = (p, s) => iprettify(extname(p)?.slice?.(1), s);
const save = (p, s) => p !== "null" && ensureDir(p) && writeFileSync(p, prettify(p, s), 'utf-8');

const getData = (data, env, name) => {
	if (typeof data === "object") {
		if (data.type === 'object')
			return fromObject({...env, ...data, [name]: undefined});
		if (data.type === 'link')
			return fromObject({...data, ...schema[getData(data.value)]});
		if (data.type === 'file')
			return fromFile(getData(data.value), data);
		if (data.type === 'olink') {
			return {...data, ...schema[data.value]};
		}
		return data
	} else return data;
};

const build = (s, data) => {
	!ignored && console.log("\t".repeat(depth), "build", data.filename || data.value);
	for (const key in data) {
		depth += 1;
		const entry = data[key];
		const placeholder = `{{${key}(\\[[^]*\\])?}}`;
		const resolved = getData(entry, data, key);
		!ignored && console.log(resolved)
		!ignored && console.log("\t".repeat(depth), "key", key);
		s = s.replace(new RegExp(placeholder, 'gm'), (m, c) => c ? eval(c.slice(1, -1)) : data.code ? eval(data.code) : resolved);
		depth -= 1;
	}
	if (data.output && data.output !== "null") {
		!ignored && console.log("\t".repeat(depth), "save", join("dist", data.output !== "null" ? data.output || data.filename : "null"), s);
		save(join("dist", data.output !== "null" ? data.output || data.filename : "null"), s, 'utf-8');
	}
	return s;
};

const fromObject = obj => build(obj.filename ? readFileSync(join(templates, obj.filename)).toString() : obj.template || "", obj);
const fromFile = (path, data) => build(readFileSync(join(templates, path)).toString(), data || {}).toString();

if (eval_tag) {
	console.log("Evaluating", eval_tag)
	console.log(fromObject(schema[eval_tag]))
} else {
	ensureDir(output_path)
	schema.pages.forEach(page => fromObject(schema[page]))
	console.log("Build done")
}
