importScripts('./lib/tfjs/tfjs@0.11.2.js');

run()

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
    // renderChart(
    //     x_ranges // x domain range
    //     ,-1 // iteration - 1
    //     ,x_ranges.map(function(x){ // y_pred
    //         return Number(W.dataSync()[0]*x+b.dataSync()[0]).toFixed(4)
    //     }) 
    //     ,W.dataSync()[0] // W_pred
    //     ,b.dataSync()[0] // b_pred
    // )
    window.postMessage({command:"render"})

    return;
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