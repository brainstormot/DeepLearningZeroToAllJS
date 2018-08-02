const gulp = require('gulp')
const path = require('path')
const fs = require('fs');
const browserify = require('gulp-browserify');

const baseDir = __dirname
const srcPath = path.join(baseDir,"./src")

gulp.task('list:html', function () {
    console.log("list:html")
    console.log(baseDir)
    fs.readdir(srcPath,(err,labs)=>{
        labs.forEach((lab)=>{
            let labPath = path.join(srcPath,lab)
            fs.readdir(labPath,(err,pages)=>{
                pages.forEach((page)=>{
                    let pagePath = path.join(labPath,page)
                    fs.readdir(pagePath,(err,files)=>{
                        console.log(files)
                    })
                })
            })
        })
    })
});

async function browserifyTypescript(filename,src,dist){

}
