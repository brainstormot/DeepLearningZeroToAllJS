// alert(typeof tf);
var seed = 777

var x_train = [1, 2, 3]
var y_train = [1, 2, 3]

var tensor_x_train = tf.tensor1d(x_train)
var tensor_y_train = tf.tensor1d(y_train)

// log(tf.randomNormal([1],0,1,'float32',seed).print())
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

// using tf.tidy to avoid memory leak
// In tfjs, functional expression seems default.

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

const learning_rate=0.01
optimizer = tf.train.sgd(learning_rate)
// train = optimizer.minimize(cost)

// for d3 visualization

var x_ranges = _.range(_.min(x_train),_.max(x_train),0.01)
// log(x_ranges)

for (let i = 0; i < 2000; i++) {
    optimizer.minimize(()=>loss(predict(tensor_x_train),tensor_y_train));
    if(i%1000==0){
        log(`[iter ${i+1}] loss : ${loss(predict(tensor_x_train),tensor_y_train).dataSync()}`)
        let W_pred = W.dataSync();
        let b_pred = b.dataSync();
        let y_preds = x_ranges.map(function(x){return W_pred*x+b_pred})
        
        let vis = d3.select("section").append("svg");
        let line = d3.line()
                    .x(function(d) { return x_ranges[d]; })
                    .y(function(d) { return y_preds[d]; })
        let w = 700,h = 700;
        vis.attr("width", w)
            .attr("height", h);
    }
}

log(`W: ${W.dataSync()}, b: ${b.dataSync()}`)
// alert('HI')
