const inputString = "{{key}} {{key[data]}}"

const regex = /{{key(\[.*\])?}}/g;

const result = inputString.replace(regex, (match, code) => {
	let t = 2;
	return code ? eval(`(data => ${code.slice(1, -1)})("asdf")`) : "asdf"
});

console.log(result); // "This is 4 and 15."

