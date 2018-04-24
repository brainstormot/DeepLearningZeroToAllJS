// dependency on tensorflow.js ^0.9.0 and underscore ^1.9.0



function recursiveSplit(array,shapeObj){
    if(shapeObj && shapeObj.shape.length > shapeObj.index){
        return _.chain(array).chunk(shapeObj.shape[shapeObj.index++]).map(childArray=>recursiveSplit(childArray,shapeObj)).value()
    }else{
        return array
    }
}
/**
 * 
 * @param {object} tensor 
 */
function ToStackedArray(tensor){
    let shapeObj = {
        shape:_.chain(tensor.shape).initial().value()
        ,index: 0
    }
    console.log(shapeObj)
    return recursiveSplit(tensor.dataSync(),shapeObj)
}