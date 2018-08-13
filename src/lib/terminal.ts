// courtesy of ahmed-musallam  'https://gist.github.com/ahmed-musallam/d0378d3494744d412cb7b69a3313e2da'
var footer:HTMLElement = document.createElement('footer');
var terminal:HTMLElement = document.createElement('ul')
terminal.setAttribute("id", "terminal");
footer.appendChild(terminal)
document.body.appendChild(footer);
// var parentElem = consoleElem.parentElement;
// var isLoggig = true;

// const logBuffer = [];
export function log(txt:string | object,toEnd=false) {
	var newLine = document.createElement('li');
	newLine.innerHTML = typeof txt === 'string' ? txt : JSON.stringify(txt, null, 4);
	terminal.appendChild(newLine);
	if(toEnd===true)
		footer.scrollTo(0,footer.scrollHeight);
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
