import * as d3 from 'd3';
// import * as c3 from 'c3';
import * as tf from '@tensorflow/tfjs';
// import * as nj from 'numjs';
import * as _ from 'lodash';
import {log} from '../../lib/terminal';

log("backend : "+ tf.getBackend())

const DATA_FILE = "../../data/data-03-diabetes.csv"
const runButton:HTMLElement = document.getElementById('run')

interface Inputs{
    maxEpoch:number;
    printInterval:number;
    learningRate:number;
    x_data:number[][];
    y_data:number[][];
}

runButton.onclick = async ()=>{
    let text:string = await d3.text(DATA_FILE);
    const rows:string[][] = d3.csvParseRows(text)
    // console.log("rows",rows);
    const x_data = _.chain(rows).map(function(row:string[]){
        return _.chain(row).initial().map(x=>Number(x)).value()
    }).value();
    const y_data =  _.chain(rows).map(function(row){
        return _.chain(row).last().map(x=>Number(x)).value()
    }).value();

    console.log(x_data)
    console.log(y_data)

    let inputs:Inputs = {
        maxEpoch:Number((document.getElementById('maxEpoch') as HTMLInputElement).value)
        , printInterval : Number((document.getElementById('printInterval') as HTMLInputElement).value)
        , learningRate : Number((document.getElementById('learningRate') as HTMLInputElement).value)
        ,x_data:x_data
        ,y_data:y_data
    }
    console.log(inputs)
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
    tf.disposeVariables();

    var x_train = tf.tensor2d(inputs.x_data)
    var y_train = tf.tensor2d(inputs.y_data)
    log(`x_train : ${x_train}`)
    log(`y_train : ${y_train}`)

    var W:tf.Tensor2D = tf.variable(
            tf.randomNormal([8,1],0,1,'float32') 
            , true 
            , 'weight' 
            , 'float32' 
            )
    var b:tf.Tensor1D = tf.variable(
        tf.randomNormal([1],0,1,'float32')
        , true 
        , 'bias' 
        , 'float32'
        )

    log(`init W : ${W.dataSync()}`)
    log(`init b : ${b.dataSync()}`)

    function hypothesis(x:tf.Tensor2D):tf.Tensor2D{
        return tf.tidy(() => {
            return x.matMul(W).add(b).sigmoid() as tf.Tensor2D
        });
    }

    function loss(hypothesis:tf.Tensor2D, label:tf.Tensor2D):tf.Scalar{
        return tf.tidy(() => {
            // cross entropy
            return tf.add(
                label.mul(hypothesis.log())
                , tf.scalar(1).sub(label).mul(tf.scalar(1).sub(hypothesis).log())
            ).mean().neg().squeeze()
        });
    }

    function accuracy(pred:tf.Tensor2D, label:tf.Tensor2D):tf.Scalar{
        return tf.equal(pred,label).cast("float32").mean();
    }

    const optimizer = tf.train.sgd(inputs.learningRate)

    let _hypo:tf.Tensor2D = hypothesis(x_train)
    log(`init loss : ${loss(_hypo,y_train)}`)
    log(`init hypothesis : ${_hypo}`)
    log(`init prediction : ${_hypo.round()}`)
    log(`init accuracy : ${accuracy(_hypo.round(),y_train)}`)

    await tf.nextFrame()
        
    for (let i = 1; i <= inputs.maxEpoch; i++) {
        optimizer.minimize(()=>loss(hypothesis(x_train),y_train));
        if(i % inputs.printInterval==0 || inputs.maxEpoch === i){
            let _hypo:tf.Tensor2D = hypothesis(x_train)
            log(`[iter ${i}] loss : ${loss(_hypo,y_train)}`)
            log(`[iter ${i}] prediction : ${_hypo.round()}`)
            log(`[iter ${i}] accuracy : ${accuracy(_hypo.round(),y_train)}`)
            await tf.nextFrame()
        }
    }
}