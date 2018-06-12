// var seed = 777
console.log("backend : ",tf.getBackend())
const runButton = document.getElementById('run')

const x_train = [1, 2, 3]

// using d3 and underscore for generating x_ranges, but i believe there should be a better way.
// x range : [-max(x_train),max(x_train)]
// for rendering chart.
const x_ranges = _.range(101).map(
    x=>d3.scaleLinear().domain([0,100]).range([_.max(x_train)*-1,_.max(x_train)])(x)
)
const maxAbsXRange = Math.ceil(_.max(x_ranges.map(value=>Math.abs(value)))) 

const arrayCharts = []

var trueFunctionName = "y = 2*x+1"
var trueFunction;
var maxEpoch;
var printInterval;
var learningRate;

var y_train;
var y_true;

var worker

runButton.onclick = function(){
    var a = Number(document.getElementsByName('a')[0].value)
    var b = Number(document.getElementsByName('b')[0].value)
    console.log(a)
    maxEpoch = Number(document.getElementsByName('maxEpoch')[0].value)
    printInterval = Number(document.getElementsByName('printInterval')[0].value)
    learningRate = Number(document.getElementsByName('learningRate')[0].value)
    trueFunctionName = `y = ${a}*x+${b}`
    trueFunction = function(x){
        return a*x+b
    }
    y_train = _.chain(x_train).map(trueFunction).value()
    y_true = _.chain(x_ranges).map(trueFunction).value()
    if(window.Worker){
        console.log("worker available.")
        worker = new Worker("lab-02-1-linear_regression.worker.js")
        worker.onmessage = function (event) {
            console.log("Worker said : ",event);
        };
    }else{
        alert("Web Worker is not available")
        console.log("Web Worker is not available")
    }
    // run();
}



function init(){
    var slider = document.getElementById("epochRange");
    slider.max = Math.floor(maxEpoch/printInterval) + 1;
    slider.min = 0;
    var output = document.getElementById("epochValue");
    makeTextHTML();
    slider.oninput = function() {
        makeTextHTML();
    }

    function makeTextHTML(){
        if(slider.max === slider.value){
            output.innerHTML = `Final result (epoch : ${maxEpoch})`;            
        }else{
            output.innerHTML = `epoch : ${slider.value*printInterval}`;
        }
    }
}

function renderChart(x_ranges,i,y_preds,W_pred,b_pred){
    let output_2 = document.getElementById('output_2') 
    let newDiv = document.createElement('div')
    newDiv.setAttribute("id", `iter${i+1}`)
    // newDiv.setAttribute("style", `inline-block`)
    newDiv.style = "display:inline-block;width:340px;"
    // newDiv.setAttribute("margin", `inline-block`)
    // newDiv.setAttribute("width", `400px`)
    output_2.appendChild(newDiv)
    
    //using billboard.js
    var chart = bb.generate({
        "data": {
            x:"x"
            ,"columns": [
                ["x",...x_ranges]
                ,[`[iter ${i}] y = ${W_pred.toFixed(3)}*x + (${b_pred.toFixed(3)})`,...y_preds]
                ,[`[true] ${trueFunctionName}`, ...y_true]
             ]
        }
        ,size: {
            "height": 300,
            "width": 320,
        },
        axis: {
            x: {
                tick: {
                    count: 11
                    ,culling: false
                    ,outer: false
                    ,format: function(x) {
                        return x.toFixed(1);
                    }
                  }                  
                ,min: maxAbsXRange*-1
                ,max: maxAbsXRange
                ,padding: {
                    right: 0,
                    left: 0
                }
            },
            y:{
                tick: {
                    count: 11
                    ,format: function(x) {
                        return x.toFixed(1);
                    }
                  }                  
                ,min: maxAbsXRange*-1
                ,max: maxAbsXRange
                ,padding: {
                    top: 0,
                    bottom: 0
                }
            }
        },
        point: {
            show: false
        },
        legend: {
            position: "inset"
        }
        ,grid: {
            x: {
              show: true,
              lines: [
                {value: 0, text: ""},
              ]
            },
            y: {
              show: true,
              lines: [
                {value: 0, text: ""}
              ]
            },
            focus: {
               show: false
            },
            lines: {
               front: false
            }
          }
        ,bindto: `#iter${i+1}`
    });
}