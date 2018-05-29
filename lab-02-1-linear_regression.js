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
    document.getElementsByClassName('tablinks')[0].click();
    run();
}




// main function
function run(){
    tf.disposeVariables ()
    init()
    var tensor_x_train = tf.tensor1d(x_train)
    var tensor_y_train = tf.tensor1d(y_train)

    var W = tf.variable(
            tf.randomNormal([1],0,1,'float32') // values
            , true // trainable 
            , 'weight' // name
            , 'float32' // dtype
            )
    var b = tf.variable(
        tf.randomNormal([1],0,1,'float32') // values
        , true // trainable 
        , 'bias' // name
        , 'float32' // dtype
        )
    log(`init W : ${W.dataSync()}`)
    log(`init b : ${b.dataSync()}`)

    // using tf.tidy to avoid memory leak. But, in this case, I suppose that tf.tidy is not necessary
    function predict(x){
        return tf.tidy(() => {
            return x.mul(W).add(b)
        });
    }

    function loss(pred, label){
        return tf.tidy(() => {
            return pred.sub(label).square().mean();
        });
    }

    optimizer = tf.train.sgd(learningRate)

    // before training
    renderChart(
        x_ranges // x domain range
        ,-1 // iteration - 1
        ,x_ranges.map(function(x){ // y_pred
            return Number(W.dataSync()[0]*x+b.dataSync()[0]).toFixed(4)
        }) 
        ,W.dataSync()[0] // W_pred
        ,b.dataSync()[0] // b_pred
    )

    for (let i = 1; i <= maxEpoch; i++) {
        optimizer.minimize(()=>loss(predict(tensor_x_train),tensor_y_train));
        if(i%printInterval==0){
            log(`[iter ${i+1}] loss : ${loss(predict(tensor_x_train),tensor_y_train).dataSync()}`)
            let W_pred = W.dataSync()[0];
            // console.log(W_pred)
            let b_pred = b.dataSync()[0];
            // console.log(b_pred)        
            let y_preds = x_ranges.map(function(x){
                return Number(W_pred*x+b_pred).toFixed(4)
            })
            // console.log(["y_preds", ...y_preds])
            renderChart(x_ranges,i,y_preds,W_pred,b_pred)

        }
    }

    // after training
    log(`W: ${W.dataSync()}, b: ${b.dataSync()}`)
    renderChart(
        x_ranges
        ,maxEpoch
        ,x_ranges.map(function(x){
            return Number(W.dataSync()[0]*x+b.dataSync()[0]).toFixed(4)
        })
        ,W.dataSync()[0]
        ,b.dataSync()[0]
    )
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

function openTab(evt,id){
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    document.getElementById(id).style.display = "block";
    evt.currentTarget.className += " active";
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