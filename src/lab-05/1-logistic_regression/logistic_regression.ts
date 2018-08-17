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
    x_data:number[][]
    y_data:number[][]
}


const getX = (_string:string)=>{
    var regex2Darray = /\[\s*(\d+)\s*,\s*(\d+)\s*\]/g;
    let new_array:number[][] = [];
    _string.replace(regex2Darray, ($0:string, $1:any, $2:any):string => {
        new_array.push([Number($1), Number($2)]);
        return ""
    })
    return new_array;
}

const getY = (_string:string)=>{
    var regex2Darray = /\[\s*(\d+)\s*]/g;
    let new_array:number[][] = [];
    _string.replace(regex2Darray, ($0:string, $1:any):string => {
        new_array.push([Number($1)]);
        return ""
    })
    return new_array;
}
runButton.onclick = ()=>{

    let inputs:Inputs = {
        maxEpoch:Number((document.getElementById('maxEpoch') as HTMLInputElement).value)
        , printInterval : Number((document.getElementById('printInterval') as HTMLInputElement).value)
        , learningRate : Number((document.getElementById('learningRate') as HTMLInputElement).value)
        , x_data : getX((document.getElementById('x_data') as HTMLInputElement).value)
        , y_data:getY((document.getElementById('y_data') as HTMLInputElement).value)
    }
    if(inputs.y_data.length !== inputs.x_data.length){
        alert("x data와 y data 의 길이가 맞지 않습니다.")
        return;
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
            tf.randomNormal([2,1],0,1,'float32') 
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

    log(`learning_rate : ${inputs.learningRate}`)
    const optimizer = tf.train.sgd(inputs.learningRate)


    let _hypo:tf.Tensor2D = hypothesis(x_train)
    log(`init loss : ${loss(_hypo,y_train)}`)
    log(`init hypothesis : ${_hypo}`)
    log(`init prediction : ${_hypo.round()}`)
    log(`init accuracy : ${accuracy(_hypo.round(),y_train)}`)

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
