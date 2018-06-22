function main(){    
    d3.text("./data-03-diabetes.csv", async function(text) {
        rows = d3.csvParseRows(text)
        // console.log(rows);

        const x_data = _.chain(rows).map(function(row){
            return _.chain(row).first(8).map(x=>Number(x)).value()
        }).value()
        const y_data =  _.chain(rows).map(function(row){
            return _.chain(row).last().map(x=>Number(x)).value()
        }).value()
        // console.log(x_data)
        // console.log(y_data)

        const maxEpoch = 4001
        const printInterval = 500
        
        var x_train = tf.tensor2d(x_data)
        var y_train = tf.tensor2d(y_data)
        log(`x_train : ${x_train}`)
        log(`y_train : ${y_train}`)
    
        var W = tf.variable(
                tf.randomNormal([8,1],0,1,'float32') 
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
                ).mean().neg().squeeze()
            });
        }
    
        const learning_rate=0.01
        log(`learning_rate : ${learning_rate}`)

        await tf.nextFrame()
            
        optimizer = tf.train.sgd(learning_rate)
    
        /**
         *  @param {object} predict Tensor[x_data.legnth ,1]
         *  @return {object} number[x_data.legnth ,1] : predicted label 
         */
        function predicted(predict){
            return _.chain(predict.dataSync()).map(function(value){
                return value > 0.5 ? [1] : [0]; 
            }).value()
        }
    
        for (let i = 0; i <= maxEpoch; i++) {
            optimizer.minimize(()=>loss(predict(x_train),y_train));
            if(i%printInterval==0){
                log(`[iter ${i+1}] loss : ${loss(predict(x_train),y_train)}`)
                log(`[iter ${i+1}] Accuracy : ${tfUtils.accuracy(predicted(predict(x_train)),y_data)}`)
                await tf.nextFrame()
            }
        }
    
        // after training
        log(`[final result] loss : ${loss(predict(x_train),y_train)}`)
        log(`[final result] Accuracy : ${tfUtils.accuracy(predicted(predict(x_train)),y_data)}`)
    });
}
