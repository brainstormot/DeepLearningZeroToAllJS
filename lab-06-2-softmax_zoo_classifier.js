const nb_classes = 7 // 0 ~ 6
const num_w = 16

const maxEpoch = 2001
const printInterval = 100
const learning_rate=0.1


// main function
function main(){
    d3.text("./data-04-zoo.csv",async function(text) {
        rows = d3.csvParseRows(text)
        rows = _.filter(rows,function(row){
            return !row[0].startsWith('#')
        })
        // console.log(rows);

        const x_data = _.chain(rows).map(function(row){
            return _.chain(row).first(num_w).map(x=>Number(x)).value()
        }).value()
        // animal type
        const y_data =  _.chain(rows).map(function(row){
            return _.chain(row).last().map(x=>Number(x)).value()
        }).value()
        // console.log(x_data)
        // console.log(y_data)
        var x_train = tf.tensor2d(x_data)
        var y_train = tf.tensor2d(y_data)
        var y_one_hot = toOneHot(y_train,nb_classes)
        log(`x_train : ${x_train}`)
        log(`x_train : ${x_train}`)
        log(`y_one_hot : ${y_one_hot}`)

        var W = tf.variable(
                tf.randomNormal([num_w,nb_classes],0,1,'float32') 
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

        log(`init W : ${ToStackedArray(W,3)}`)
        log(`init b : ${ToStackedArray(b,3)}`)
        log(`learning_rate : ${learning_rate}`) 

        await tf.nextFrame()

        function predict(x){
            return tf.tidy(() => {
                return x.matMul(W).add(b).softmax()
            });
        }

        function loss(pred, label){
            // console.log(pred.shape)
            // console.log(label.shape)
            return tf.tidy(() => {
                // cross entropy with softmax
                return label.mul(pred.log()).sum(1).mean().mul(tf.tensor1d([-1])).squeeze()
            });
        }

            
        optimizer = tf.train.sgd(learning_rate)

        /**
         *  @param {object} predict Tensor
         *  @return {object} number[x_data.legnth] : predicted label 
         */
        function predicted(predict_Tensor){
            let predict  = ToStackedArray(predict_Tensor) // dataSync() and data() returns flatten data.
            let indexes = _.chain(predict).map((row)=>argMax(row)).value()
            return indexes;
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

        var i=0;
        for (; i <= maxEpoch; i++) {
            optimizer.minimize(()=>loss(predict(x_train),y_one_hot));
            if(i%printInterval==0 || i==maxEpoch){
                log(`[iter ${String(i+1).padStart(4,0)}] loss : ${Number(loss(predict(x_train),y_one_hot).dataSync()).toFixed(3)}  Accuracy : ${accuracy(predicted(predict(x_train)),_.flatten(y_data)).toFixed(3)}`)
                await tf.nextFrame()
            }
        }

        _.zip(predicted(predict(x_train)),_.flatten(y_data)).forEach((zipped)=>{
            log(`[${zipped[0]===zipped[1]}] Prediction: ${zipped[0]} True Y: ${zipped[1]}`)
        })
    })
}
