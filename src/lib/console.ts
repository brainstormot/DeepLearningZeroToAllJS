// courtesy of ahmed-musallam  'https://gist.github.com/ahmed-musallam/d0378d3494744d412cb7b69a3313e2da'
var consoleElem = document.querySelector('#console')
var parentElem = consoleElem.parentElement;
// var isLoggig = true;

// const logBuffer = [];
export function log(txt:string | object,toEnd=false) {
	var newLine = document.createElement('li');
	newLine.innerHTML = typeof txt === 'string' ? txt : JSON.stringify(txt, null, 4);
	consoleElem.appendChild(newLine);
	if(toEnd===true)
		parentElem.scrollTo(0,parentElem.scrollHeight);
		// parentElem.scrollTop = parentElem.scrollHeight;
}

// function logAsync(txt, toEnd=false){
// 	var newLine = document.createElement('li');
// 	newLine.innerHTML = typeof txt === 'string' ? txt : JSON.stringify(txt, null, 4);
// 	logBuffer.push(newLine);
// }

// var timer = setInterval(showLogs, 100);

// function showLogs(){
// 	if(isLoggig){
		
// 	}
// }
