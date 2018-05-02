// performance-01.js and results
var foo = [];
var N = 5;
var iter = 1000;

for (var i = 1; i <= N; i++) {
   foo.push(i);
}

var a = tf.tensor(
    foo
)
var b = tf.tensor(
    foo
)

console.log(`N : ${N}`)
console.log(`iter : ${iter}`)

function main1(){
    var t0 = performance.now();    
    for(let i=0;i<iter;i++){
        var c = a.equal(b).dataSync()
    }
    var t1 = performance.now();
    console.log("Call to main1 took " + (t1 - t0) + " milliseconds.")
}
function main2(){
    var t0 = performance.now();
    for(let i=0;i<iter;i++){
        var b_array = b.dataSync()
        a.dataSync().map(function(item, index) {
            return item === b_array[index];
        })
    }
    var t1 = performance.now();
    console.log("Call to main2 took " + (t1 - t0) + " milliseconds.")
}

// 1000 iter
// N     | main1 | main2
// N=1  | 899 ms | 4.3 ms
// N=5  | 566.7 ms | 2.6 ms
// N=10  | 623 ms | 3.4 ms
// N=50  | 580 ms | 5.2 ms
// N=100 | 797 ms | 7.6 ms
// N=500 | 636 ms | 24.7 ms
// N=1000 | 670 ms | 39 ms
// N=5000 | 1239 ms | 192 ms
// N=10000 | 1583 ms | 304 ms
// N=50000 | 2461 ms | 1592 ms
// N=100000 | 3972 ms | 3020 ms
// N=500000 | 18011 ms | 15308 ms