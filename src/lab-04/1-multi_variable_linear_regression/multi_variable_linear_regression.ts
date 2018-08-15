// import * as c3 from 'c3';
import * as tf from '@tensorflow/tfjs';
// import * as nj from 'numjs';
import {log} from '../../lib/terminal';

log("backend : "+ tf.getBackend())

const runButton:HTMLElement = document.getElementById('run')

interface Inputs{
    maxEpoch:number;
    printInterval:number;
    learningRate:number;
    x_datas:number[][]
    y_data:number[]
}

runButton.onclick = ()=>{
    let inputs:Inputs = {
        maxEpoch:Number((document.getElementById('maxEpoch') as HTMLInputElement).value)
        , printInterval : Number((document.getElementById('printInterval') as HTMLInputElement).value)
        , learningRate : Number((document.getElementById('learningRate') as HTMLInputElement).value)
        , x_datas : [
            (document.getElementById('x1') as HTMLInputElement).value.split(",").map(x=> Number(x))
            ,(document.getElementById('x2') as HTMLInputElement).value.split(",").map(x=> Number(x))
            ,(document.getElementById('x3') as HTMLInputElement).value.split(",").map(x=> Number(x))
        ]
        , y_data:(document.getElementById('y') as HTMLInputElement).value.split(",").map(x=> Number(x))
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

    const x_train = tf.tensor2d(inputs.x_datas)
    const y_train = tf.tensor1d(inputs.y_data)

    var W:tf.Tensor2D = tf.variable(
            tf.randomNormal([3,1],0,1,'float32') 
            , true 
            , 'weight1' 
            , 'float32' 
            )
    var b:tf.Scalar = tf.variable(
        tf.randomNormal([],0,1,'float32')
        , true 
        , 'bias' 
        , 'float32'
        )

    log(`init W : [${Array.from(W.dataSync()).map(x=>x.toPrecision(4))}]`)
    log(`init b : ${Number(b.dataSync()).toPrecision(4)}`)

    function predict():tf.Tensor<tf.Rank.R1>{
        return tf.tidy(() => {
            return x_train.mul(W).add(b).sum(0)
        });
    }

    log(`init prediction : [${Array.from(predict().dataSync()).map(x=>x.toPrecision(4))}]`)


    function loss(pred:tf.Tensor1D, label:tf.Tensor1D):tf.Scalar{
        return tf.tidy(() => {
            return pred.sub(label).square().mean();
        });
    }

    log(`init loss : ${Number(loss(predict(),y_train).dataSync()).toPrecision(4)}`)
    
    const optimizer = tf.train.sgd(inputs.learningRate)

    for (let i = 1; i <= inputs.maxEpoch; i++) {
        optimizer.minimize(()=>loss(predict(),y_train));
        if(i % inputs.printInterval === 0){
            log(`[iter ${i}] loss : ${Number(loss(predict(),y_train).dataSync()).toPrecision(4)}`)
            log(`[iter ${i}] Prediction:: [${Array.from(predict().dataSync()).map(x=>x.toPrecision(4))}]`)
            await tf.nextFrame();
        }
    }

    // after training
    log(`W : [${Array.from(W.dataSync()).map(x=>x.toPrecision(4))}], b : ${Number(b.dataSync()).toPrecision(4)}`)
}