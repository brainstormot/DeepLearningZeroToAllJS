const x_data = [[1, 2, 1],
                [1, 3, 2],
                [1, 3, 4],
                [1, 5, 5],
                [1, 7, 5],
                [1, 2, 5],
                [1, 6, 6],
                [1, 7, 7]]

const y_data = [[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0]]

const x_test = [[2, 1, 1],
                [3, 1, 2],
                [3, 3, 4]]

const y_test = [[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]]

const nb_classes = 3

const maxEpoch = 201
const printInterval = 1
const learning_rate=0.1

// main function
async function main(){
    var x_train_tensor = tf.tensor2d(x_data)
    var y_train_tensor = tf.tensor2d(y_data)
    log(`x_train_tensor : ${x_train_tensor}`)
    log(`y_train_tensor : ${y_train_tensor}`)

    var x_test_tensor = tf.tensor2d(x_test)

    var W = tf.variable(
            tf.randomNormal([3,nb_classes],0,1,'float32') 
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
            console.log("pred",ToStackedArray(pred))
            console.log("label",ToStackedArray(label))
            tf.losses.softmaxCrossEntropy(pred,label,-1).mean().print();
            return tf.losses.softmaxCrossEntropy(pred,label,1).mean();
        })
    }

       
    optimizer = tf.train.sgd(learning_rate)


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
        optimizer.minimize(()=>loss(predict(x_train_tensor),y_train_tensor));
        if(i%printInterval==0 || i==maxEpoch){
            log(`[iter ${String(i+1).padStart(4,0)}] loss : ${Number(loss(predict(x_train_tensor),y_train_tensor).dataSync()).toFixed(3)}  Accuracy : ${accuracy(predicted(predict(x_train_tensor)),y_data).toFixed(3)}`)
            await tf.nextFrame()
        }

    }
    _.zip(predicted(predict(x_test_tensor)),y_test).forEach((zipped)=>{
        log(`[${_.isEqual(zipped[0],zipped[1])}] Prediction: ${zipped[0]} True Y: ${zipped[1]}`)
    })
}
