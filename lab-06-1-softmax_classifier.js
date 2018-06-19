const x_data = [[1, 2, 1, 1],
                [2, 1, 3, 2],
                [3, 1, 3, 4],
                [4, 1, 5, 5],
                [1, 7, 5, 5],
                [1, 2, 5, 6],
                [1, 6, 6, 6],
                [1, 7, 7, 7]]

const y_data = [[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0]]

const nb_classes = 3

const maxEpoch = 2001
const printInterval = 200
const learning_rate=0.1

// main function
async function main(){
    var x_train = tf.tensor2d(x_data)
    var y_train = tf.tensor2d(y_data)
    log(`x_train : ${x_train}`)
    log(`y_train : ${y_train}`)

    var W = tf.variable(
            tf.randomNormal([4,nb_classes],0,1,'float32') 
            , true 
            , 'weight' 
            , 'float32' 
            )
    var b = tf.variable(
        tf.randomNormal([nb_classes],0,1,'float32')
        , true 
        , 'bias' 
        , 'float32'
        )

    log(`init W : ${ToStackedArray(W)}`)
    log(`init b : ${ToStackedArray(b)}`)
    log(`learning_rate : ${learning_rate}`) 

    await tf.nextFrame()

    function predict(x){
        return tf.tidy(() => {
            return x.matMul(W).add(b).softmax()
        });
    }

    function loss(pred, label){
        return tf.tidy(() => {
            // cross entropy with softmax
            return label.mul(pred.log()).sum(1).mean().neg().squeeze()
        });
    }

       
    optimizer = tf.train.sgd(learning_rate)

    /**
     *  @param {object} predict Tensor[x_data.legnth ,1]
     *  @return {object} number[x_data.legnth ,1] : predicted label 
     */
    function predicted(predict_Tensor){
        let predict  = _.chunk(predict_Tensor.dataSync(),nb_classes) // dataSync() and data() returns flatten data.
        let oneHotVectors = _.chain(predict).map(
            function(row){
                let oneHotVector = Array.apply(null, Array(nb_classes)).map(Number.prototype.valueOf,0)
                let index = _.chain(row).reduce(function(prev,current,index){
                    if(prev.value < current){
                        prev.value = current;
                        prev.index = index;
                    }
                    return prev
                    },{index:-1,value:-1}).value().index
                oneHotVector[index] = 1;
                return oneHotVector
            }
        ).value()
        return oneHotVectors;
    }
    /**
     * @param { number[x_data.legnth ,1] } predicted_labels
     * @param { number[x_data.legnth ,1] } true_labels
     * @return { number } accuracy 
     */
    function accuracy(predicted_labels,true_labels){
        let matchCount= _.chain(_.zip(predicted_labels,true_labels)).reduce(function(sum,pair){
            if(_.isEqual(pair[0],pair[1])){
                return sum+1;
            }else{
                return sum;
            }
        },0).value()
        return matchCount / predicted_labels.length
    }

    for (let i = 0; i <= maxEpoch; i++) {
        optimizer.minimize(()=>loss(predict(x_train),y_train));
        if(i%printInterval==0 || i==maxEpoch){
            log(`[iter ${String(i+1).padStart(4,0)}] loss : ${Number(loss(predict(x_train),y_train).dataSync()).toFixed(3)}  Accuracy : ${accuracy(predicted(predict(x_train)),y_data).toFixed(3)}`)
            await tf.nextFrame()
        }

    }
    _.zip(predicted(predict(x_train)),y_data).forEach((zipped)=>{
        log(`[${_.isEqual(zipped[0],zipped[1])}] Prediction: ${zipped[0]} True Y: ${zipped[1]}`)
    })
}
