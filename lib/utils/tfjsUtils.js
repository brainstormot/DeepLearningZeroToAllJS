// dependency on tensorflow.js ^0.9.0 and underscore ^1.9.0

/**
 * 
 * @param {object} tensor
 * @returns {object} stacked array
 */
function ToStackedArray(tensor){
    console.log(_.last(tensor.shape,tensor.shape.length-1))
    return _.chain(_.last(tensor.shape,tensor.shape.length-1)).reduceRight(
        function(accumulator,currentValue){
            return _.chunk(accumulator,currentValue)
        }
        ,tensor.dataSync()
    ).value()
}

