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
// x range : [-max(x_train),max(x_train)]
var x_ranges = _.range(101).map(
    x=>d3.scaleLinear().domain([0,100]).range([_.max(x_train)*-1,_.max(x_train)])(x)
)
const maxAbsXRange = Math.ceil(_.max(x_ranges.map(value=>Math.abs(value))))

console.log(maxAbsXRange)
console.log(["x",...x_ranges])

// var chart;
// chart.x(x_ranges)
// console.log("chart : ",chart.x())
 
// log(x_ranges)
var canvas = document.getElementById('canvas')

for (let i = 0; i < 2000; i++) {
    optimizer.minimize(()=>loss(predict(tensor_x_train),tensor_y_train));
    if(i%100==0){
        log(`[iter ${i+1}] loss : ${loss(predict(tensor_x_train),tensor_y_train).dataSync()}`)
        let W_pred = W.dataSync()[0];
        // console.log(W_pred)
        let b_pred = b.dataSync()[0];
        // console.log(b_pred)        
        let y_preds = x_ranges.map(function(x){
            return Number(W_pred*x+b_pred).toFixed(4)
        })
        // console.log(["y_preds", ...y_preds])
        
        let newDiv = document.createElement('div')
        newDiv.setAttribute("id", `iter${i+1}`)
        newDiv.setAttribute("width", `250px`)
        newDiv.setAttribute("height", `250px`)
        newDiv.setAttribute("display", `inline`)
        canvas.appendChild(newDiv)
        
        // const maxAbsYRange = Math.ceil(_.max(y_preds.map(value=>Math.abs(value))))
        let chart = bb.generate({
            "data": {
                x:"x"
                ,"columns": [
                    ["x",...x_ranges]
                    ,[`y_preds iter ${i+1}`,...y_preds]
                 ]
            }
            ,size: {
                "height": 200,
                "width": 200,
            },
            axis: {
                x: {
                    tick: {
                        count: 5
                      }                  
                    ,min: maxAbsXRange*-1
                    ,max: maxAbsXRange
                },
                y:{
                    tick: {
                        count: 5
                      }                  
                    ,min: maxAbsXRange*-1
                    ,max: maxAbsXRange
                }
            },
            bindto: `#iter${i+1}`
        });
    }
}

log(`W: ${W.dataSync()}, b: ${b.dataSync()}`)
// alert('HI')
