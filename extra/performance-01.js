var foo = [];
var N = 100;
var iter = 10000;

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

// 10000 iter
// N     | main1 | main2
// N=10  | 9.641 ms | main2
// N=50  | 8.317 s  | 
// N=100 | 5.691 ms | 0.31 ms