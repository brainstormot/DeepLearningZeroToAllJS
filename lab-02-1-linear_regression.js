// alert(typeof tf);
var seed = 777

var x_train = [1, 2, 3]
var y_train = [1, 2, 3]

const model = tf.sequential();

log(tf.initializers.randomNormal({mean:0,stddev:1.0,seed:seed}))
var W = tf.variable(
    tf.tensor(
        tf.initializers.randomNormal({mean:0,stddev:1.0,seed:seed}) // values
        , true // trainable 
        , 'weight' // name
        , 'float32' // dtype
        )
)  ;
// const b = tf.variable(tf.tensor([1, 2, 3]), 'bias');
log(typeof tf.initializers.randomNormal)
log(W)
log(W.print())


// for(var i=0;i<30;i++){
//     log("HI"+i)
// }

// alert('HI')
