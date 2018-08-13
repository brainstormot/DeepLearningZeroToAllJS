// import * as d3 from 'd3';
import * as c3 from 'c3';
import * as tf from '@tensorflow/tfjs';


tf.print(tf.ones([1]))
c3.generate({
    bindto: '#chart',
    data: {
        columns: [
            ['data1', 30, 200, 100, 400, 150, 250],
            ['data2', 50, 20, 10, 40, 15, 25]
        ]
    }
});

// let svg = d3.select("body").append("svg");
// const w = 960;
// const h = 480;

// svg.attr("width",w);
// svg.attr("height", h);

// let dataset = [5,10,15,20,25];
// console.log("Dataset" + dataset)

// let circle = svg.selectAll("circle")
// .data(dataset)
// .enter()
// .append("circle")

// circle.attr("cx", function(d,i){
//     return (i * 50) + 25;
// })
// .attr("cy", h/2)
// .attr("r", function(d){
//     return d;
// })


// let a:tf.Tensor = tf.zeros([1])
// console.log(a)
// console.log("HI")
// console.log(typeof tf)