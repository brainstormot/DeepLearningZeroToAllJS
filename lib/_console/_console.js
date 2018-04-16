// courtesy of ahmed-musallam  'https://gist.github.com/ahmed-musallam/d0378d3494744d412cb7b69a3313e2da'
var parentElem = document.querySelector('#console').parentElement;
function log(txt) {
	var newLine = document.createElement('li');
	newLine.innerHTML = typeof txt === 'string' ? txt : JSON.stringify(txt, null, 4);
	document.querySelector('#console').appendChild(newLine);
	parentElem.scrollTop = parentElem.scrollHeight;
}
