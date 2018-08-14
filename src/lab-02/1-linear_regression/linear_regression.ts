import * as c3 from 'c3';
import * as tf from '@tensorflow/tfjs';
import * as nj from 'numjs';
import {log} from '../../lib/terminal';

log("backend : "+ tf.getBackend())

const runButton:HTMLElement = document.getElementById('run')

interface Inputs{
    maxEpoch:number;
    printInterval:number;
    learningRate:number;
    a:number;
    b:number;
    train_x_coords:number[];
    y_limit?:number[];
    trueFunctionName():string;
    trueFunction(x:number):number;
}

runButton.onclick = ()=>{
    let inputs:Inputs = {
        maxEpoch:Number(document.getElementById('maxEpoch').getAttribute('value'))
        , printInterval : Number(document.getElementById('printInterval').getAttribute('value'))
        , learningRate : Number(document.getElementById('learningRate').getAttribute('value'))
        , a : Number(document.getElementById('variable_a').getAttribute('value'))
        , b : Number(document.getElementById('variable_b').getAttribute('value'))
        , train_x_coords: (document.getElementById('train_x_coords') as HTMLInputElement).value.split(",").map(x=> Number(x)).filter(x=>!isNaN(x))
        , trueFunctionName : ()=>{
            return `y = ${inputs.a}*x+${inputs.b}`
        }
        , trueFunction : (x:number)=>{
            return inputs.a * x + inputs.b
        }
    }
    console.log(inputs)
    
    // y_train = _.chain(x_train).map(trueFunction).value()
    // y_true = _.chain(x_ranges).map(trueFunction).value()
    run(inputs);
}

function clean(){
    let graphs = document.getElementById('graphs')
    while (graphs.firstChild) {
        graphs.removeChild(graphs.firstChild);
    }
}

async function run(inputs:Inputs){
    clean();
    tf.disposeVariables ();

    const x_coords:number[] = nj.arange(Math.min(...inputs.train_x_coords),Math.max(...inputs.train_x_coords), 0.05).tolist();
    const train_y_coords:number[] = inputs.train_x_coords.map(x=>inputs.trueFunction(x));
    const true_y_coords:number[] = x_coords.map(x=>inputs.trueFunction(x));

    const x_range:number[] = [Math.min(...x_coords),Math.max(...x_coords)]
    const y_range:number[] = [Math.min(...true_y_coords),Math.max(...true_y_coords)]
    
    console.log("x_coords",x_coords)
    log(`true_x_range : ${x_range}`)
    log(`true_y_range : ${y_range}`)

    // const tensor_x_train = tf.tensor1d(inputs.train_x_coords)
    // const tensor_y_train = tf.tensor1d(train_y_coords);

    var W:tf.Tensor1D = tf.variable(
        tf.randomNormal([1],0,1,'float32') // values
        , true // trainable 
        , 'weight' // name
        , 'float32' // dtype
        )
    var b:tf.Tensor1D = tf.variable(
        tf.randomNormal([1],0,1,'float32') // values
        , true // trainable 
        , 'bias' // name
        , 'float32' // dtype
        )
    log(`init W : ${W.dataSync()}`)
    log(`init b : ${b.dataSync()}`)

    // function predict(x:tf.Tensor1D):tf.Tensor1D{
    //     return tf.tidy(() => {
    //         return x.mul(W).add(b)
    //     });
    // }

    // function loss(pred:tf.Tensor1D, label:tf.Tensor1D):tf.Tensor1D{
    //     return tf.tidy(() => {
    //         return pred.sub(label).square().mean();
    //     });
    // }

    // const optimizer = tf.train.sgd(inputs.learningRate)

    renderChart({
        true_function_name:inputs.trueFunctionName()
        ,iter:0
        ,x_coords:x_coords
        ,true_y_coords:true_y_coords
        ,train_x_coords:inputs.train_x_coords
        ,train_y_coords:train_y_coords
        ,x_range:x_range
        ,y_range:y_range
        ,W:W.dataSync()[0]
        ,b:b.dataSync()[0]
    });
}

interface ChartInput{
    true_function_name:string;
    iter:number;
    x_coords:number[];
    true_y_coords:number[];
    train_x_coords:number[];
    train_y_coords:number[];
    x_range:number[];
    y_range:number[];
    W:number;
    b:number;
}

function generateChartConfiguration(chartInput:ChartInput):c3.ChartConfiguration{
    const pred_y_coords = chartInput.x_coords.map(x=>chartInput.W*x + chartInput.b)
    return {
        bindto: `#iter${chartInput.iter}`
        ,data:{
            xs:{
                true_y:'x'
                ,pred_y:'x'
                ,train_y:'train_x'
            }
            ,columns:[
                ['x', ...chartInput.x_coords]
                ,['true_y',...chartInput.true_y_coords]
                ,['pred_y',...pred_y_coords]
                ,['train_x', ...chartInput.train_x_coords]
                ,['train_y', ...chartInput.train_y_coords]
            ]
            ,types:{
                true_y:'line'
                ,pred_y:'line'
                ,train_y:'scatter'
            }
            ,names:{
                true_y: `[True] ${chartInput.true_function_name}`
                ,pred_y: `'[Pred] ${chartInput.W.toFixed(3)}*x + (${chartInput.b.toFixed(3)})`
                ,train_y: `training points`
            }
        },
        axis: {
            x: {
                label: 'x',
                tick: {
                    fit: false
                }
                ,min:chartInput.x_range[0]
                ,max:chartInput.x_range[1]
            },
            y: {
                label: 'y'
                ,min:chartInput.y_range[0]
                ,max:chartInput.y_range[1]
            }
        }
    }
}

function renderChart(chartInput:ChartInput){
    let graphs:HTMLElement = document.getElementById('graphs') 
    let newDiv:HTMLDivElement = document.createElement('div')
    newDiv.setAttribute("id", `iter${chartInput.iter}`)
    newDiv.setAttribute("style","display:inline-block;width:340px;");
    graphs.appendChild(newDiv)

    const configuartion = generateChartConfiguration(chartInput)
    c3.generate(configuartion);
}



// let svg = d3.select("body").append("svg");
// const w = 960;
// const h = 480;

// svg.attr("width",w);
// svg.attr("height", h);

// let dataset = [5,10,15,20,25];
// console.log("Dataset" + dataset)

// let circle = svg.selectAll("circle")
// .data(dataset)
// .enter()
// .append("circle")

// circle.attr("cx", function(d,i){
//     return (i * 50) + 25;
// })
// .attr("cy", h/2)
// .attr("r", function(d){
//     return d;
// })


// let a:tf.Tensor = tf.zeros([1])
// console.log(a)
// console.log("HI")
// console.log(typeof tf)