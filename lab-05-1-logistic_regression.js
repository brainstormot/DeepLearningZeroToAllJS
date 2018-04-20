// var seed = 777

const x_data = [[1, 2],
                [2, 3],
                [3, 1],
                [4, 3],
                [5, 3],
                [6, 2]]

const y_data = [[0],
                [0],
                [0],
                [1],
                [1],
                [1]]

const maxEpoch = 10001
const printInterval = 200

// main function
function main(){
    var x_train = tf.tensor2d(x_data)
    var y_train = tf.tensor2d(y_data)
    log(`x_train : ${x_train}`)
    log(`y_train : ${y_train}`)

    var W = tf.variable(
            tf.randomNormal([2,1],0,1,'float32') 
            , true 
            , 'weight' 
            , 'float32' 
            )
    var b = tf.variable(
        tf.randomNormal([1],0,1,'float32')
        , true 
        , 'bias' 
        , 'float32'
        )

    log(`init W : ${W.dataSync()}`)
    log(`init b : ${b.dataSync()}`)

    function predict(x){
        return tf.tidy(() => {
            return x.matMul(W).add(b).sigmoid()
        });
    }

    function loss(pred, label){
        return tf.tidy(() => {
            // cross entropy
            return tf.add(
                label.mul(pred.log())
                , tf.mul(tf.ones([1]).sub(label), tf.ones([1]).sub(pred).log())
            ).mean().mul(tf.tensor1d([-1])).squeeze()
        });
    }

    const learning_rate=0.01
    log(`learning_rate : ${learning_rate}`) 
       
    optimizer = tf.train.sgd(learning_rate)

    /**
     *  @param {object} predict Tensor[x_data.legnth ,1]
     *  @return {object} number[x_data.legnth ,1] : predicted label 
     */
    function predicted(predict){
        return _.chain(predict.dataSync()).map(function(value){
            return value > 0.5 ? 1 : 0; 
        }).value()
    }
    /**
     * @param { number[x_data.legnth ,1] } predicted_labels
     * @param { number[x_data.legnth ,1] } true_labels
     * @return { number } accuracy 
     */
    function accuracy(predicted_labels,true_labels){
        let matchCount= _.chain(_.zip(predicted_labels,true_labels)).reduce(function(sum,pair){
            if(pair[0]===pair[1]){
                return sum+1;
            }else{
                return sum;
            }
        },0).value()
        return matchCount / predicted_labels.length
    }

    for (let i = 0; i <= maxEpoch; i++) {
        optimizer.minimize(()=>loss(predict(x_train),y_train));
        if(i%printInterval==0){
            log(`[iter ${i+1}] loss : ${loss(predict(x_train),y_train)}`)
            log(`[iter ${i+1}] Prediction : ${predicted(predict(x_train))}`)
            log(`[iter ${i+1}] Accuracy : ${accuracy(predicted(predict(x_train)),_.flatten(y_data))}`)
        }
    }

    // after training
    log(`[final result] loss : ${loss(predict(x_train),y_train)}`)
    log(`[final result] Prediction : ${predicted(predict(x_train))}`)
    log(`[final result] Accuracy : ${accuracy(predicted(predict(x_train)),_.flatten(y_data))}`)
}
