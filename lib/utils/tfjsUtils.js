/**
 * @description util module for using tensorflow.js
 * @requires tensorflow.js ^0.11.6
 * @typedef {Object} Tensor
 */
!function(global, factory){
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
	typeof define === 'function' && define.amd ? define(['exports'], factory) :
	(factory((global.tfUtils = global.tfUtils || {})));
}(this,function(exports){

    /**
     * @author engelen
     * @description Retrieve the array key corresponding to the largest element in the array.
     * @param {number[]} array Input array
     * @return {number} Index of array element with largest value
     */
    function argMax(array) {
        return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
    }

    /**
     * @param  {number|number[]} arrayOrNumber
     * @param  {number} precision default=1
     */
    function _round(arrayOrNumber,precision=1){
        let divisor = Math.pow(10,precision)
        if(typeof arrayOrNumber === "number"){
            return Math.round(arrayOrNumber * divisor) / divisor
        }else{
            return arrayOrNumber.map((x)=>Math.round(x * divisor) / divisor)
        }
    }

    function _chunkArray(myArray, chunk_size){
        var results = [];
        
        while (myArray.length) {
            results.push(myArray.splice(0, chunk_size));
        }
        
        return results;
    }

    /**
     * @param {Tensor} tensor
     * @param {number} digit truncated digit. if digit==3, then 0.12315215 -> 0.123
     * @returns {number[]} stacked array or scalar
     */
    function toArraySync(tensor,digit=undefined){
        if(digit && typeof digit != "number"){
            digit = Number(digit)
            console.log(digit)
        }
        if(tensor.shape.length===0){return digit && digit > 0 ? _round(tensor.dataSync(),digit) : tensor.dataSync()}
        // console.log(_.last(tensor.shape,tensor.shape.length-1))
        let arr = Array.from(tensor.dataSync())
        if(digit && digit > 0) arr = _round(arr,digit)
        return tensor.shape.slice(1).reduceRight((acc,cur)=>{
            return acc = _chunkArray(acc,cur)
        },arr)
    }
    /**
     * @async
     * @param {Tensor} tensor
     * @param {number} digit truncated digit. if digit==3, then 0.12315215 -> 0.123
     * @returns {Promise<number[]>} promise which resolves stacked array or scalar
     */
    async function toArray(tensor,digit=undefined){
        if(digit && typeof digit != "number"){
            digit = Number(digit)
            console.log(digit)
        }
        if(tensor.shape.length===0){return digit && digit > 0 ? _round(await tensor.data(),digit) : await tensor.data()}
        // console.log(_.last(tensor.shape,tensor.shape.length-1))
        let arr = Array.from(await tensor.data())
        console.log(arr)

        if(digit && digit > 0) arr = _round(arr,digit)

        return tensor.shape.slice(1).reduceRight((acc,cur)=>{
            return acc = _chunkArray(acc,cur)
        },arr)
    }

    /**
     *  @description softmax normalized vectors(tensor) to one-hot vectors(tensor) Ex) [[0.9,0.1],[0.2,0.8]] -> [[1,0],[0,1]]
     *  @param {Tensor} tensor softmax normalized vector with one dimension wrapper
     *  @param {number} tensor.shape should be [numOfBatch,numOfClasses]
     *  @return {Tensor}
     */
    function toOneHotTensor(tensor){
        return tensor.argMax(1).oneHot(tensor.shape[1])
    }


    /**
     * @description description
     * @param { number[] } predicted_labels one-hot vectors
     * @param { number[] } true_labels one-hot vectors
     * @return { number } accuracy 
     */
    function accuracy(predicted_labels,true_labels){
        let count = 0;
        for(var i=0,iLen=predicted_labels.length;i<iLen;i++){
            if(predicted_labels[i].length === true_labels[i].length && predicted_labels[i].every(function(v,j) { return v === true_labels[i][j]})){
                ++count;
            }
        }
        return count / predicted_labels.length
    }

    exports.toArraySync = toArraySync
    exports.toArray = toArray
    exports.toOneHotTensor = toOneHotTensor
    exports.accuracy = accuracy
    exports.argMax = argMax
    Object.defineProperty(exports, '__esModule', { value: true });
})