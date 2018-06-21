// dependency on tensorflow.js ^0.11.6 and underscore ^1.9.0
!function(e,t){
    "object" == typeof exports && "undefined" != typeof module ? t(exports) : "function" == typeof define && define.amd ? define(["exports"], t) : t(e.tfUtil = e.tfUtil || {})
}(this,function(exports){

    /**
     * courtesy by engelen
     * Retrieve the array key corresponding to the largest element in the array.
     *
     * @param {Array.<number>} array Input array
     * @return {number} Index of array element with largest value
     */
    function argMax(array) {
        return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
    }


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
     * 
     * @param {object} tensor
     * @param {number} digit truncated digit. if digit==3, then 0.12315215 -> 0.123
     * @returns {object} stacked array or scalar
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
     * 
     * @param {object} tensor
     * @param {number} digit truncated digit. if digit==3, then 0.12315215 -> 0.123
     * @returns {object} promise which resolves stacked array or scalar
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
     *  class to one-hot (3 to [0,0,1])
     *  @param {object} tensor Tensor[x_data.legnth ,1]
     *  @return {object} number[x_data.legnth ,1] : predicted label 
     */
    function toOneHot(tensor,nb_classes){
        let predict  = _chunkArray(tensor.dataSync(),nb_classes) // dataSync() and data() returns flatten data.
        let oneHotVectors = predict.map(
            function(row){
                let oneHotVector = Array(nb_classes).fill(0)
                oneHotVector[argMax(row)] = 1;
                return oneHotVector
            }
        )
        return oneHotVectors;
    }

    /**
     *  @param {object} tensor Tensor[x_data.legnth ,1]
     *  @return {object} promise which resolve number[x_data.legnth ,1] : predicted label 
     */
    async function toOneHot(tensor,nb_classes){
        let predict  = _chunkArray(await tensor.dataSync(),nb_classes) // dataSync() and data() returns flatten data.
        let oneHotVectors = predict.map(
            function(row){
                let oneHotVector = Array(nb_classes).fill(0)
                oneHotVector[argMax(row)] = 1;
                return oneHotVector
            }
        )
        return oneHotVectors;
    }

    exports.toArraySync = toArraySync
    exports.toArray = toArray
    exports.toOneHotSync = toOneHotSync
    exports.toOneHot = toOneHot
    exports.argMax = argMax
    Object.defineProperty(exports, "__esModule", {
        value: !0
    })
})