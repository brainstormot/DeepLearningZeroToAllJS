const gulp = require('gulp')
const clean = require('gulp-clean');
const rename = require('gulp-rename');
const browserify = require('browserify')
const source = require('vinyl-source-stream')
const connect = require('gulp-connect');

const path = require('path')
const fs = require('fs');

const libConfig = require('./config/lib.config')

const baseDir = __dirname
const srcPath = path.join(baseDir,"src")
const distPath = path.join(baseDir,"dist")
const distLibPath = path.join(baseDir,"dist/lib")
const srcLibPath = path.join(baseDir,"src/lib")



gulp.task('clean',function(){
    return gulp.src(distPath, {read: false})
    .pipe(clean());
})

gulp.task('build',['clean'], build)
gulp.task('build:html', build_html);
gulp.task('build:lib',build_lib)
gulp.task('build:ts', build_ts);

gulp.task('run',['build'],function(){
    connect.server({
        name:'DeepLearningZeroToAllJS'
        ,root: 'dist',
        livereload: true
        ,port:8020
        ,debug:true
    });
})

async function build(){
    let results = await Promise.all([build_html(),build_lib(),build_ts()])
    if(results.every(value=>value===true)){
        console.log("build success")
        return true
    }else{
        console.log("build failed")
        return false
    }
}

async function build_html() {
    try{
        // console.log("list:html")
        let paths = await pathsInPage()
        let htmls = paths.filter(_path => path.extname(_path[1]) === '.html').map(_path=>[_path[0], path.join(..._path)])
        // console.log(htmls)
        await Promise.all(htmls.map(async function(html){
            gulp.src(html[1]).pipe(rename('index.html')).pipe(gulp.dest(html[0].replace(srcPath,distPath)))
        }))
        return true
    }catch(e){
        return false
    }
}

async function build_lib() {
    try{
        let filePaths = await readdir(srcLibPath);
        let tsFilePaths = filePaths.filter(_path => path.extname(_path[1]) === '.ts')
        let notTsFilePaths = filePaths.filter(_path => path.extname(_path[1]) !== '.ts');

        await Promise.all(tsFilePaths.map(async function(tsfile){
            var bundleStream = browserify(path.join(...tsfile)).plugin('tsify').bundle()
            // var bundleStream = browserify(path.join(...tsfile).plugin('tsify').bundle()
            bundleStream
                .pipe(source(tsfile[1]))
                .pipe(rename(tsfile[1].replace('.ts','.js')))
                .pipe(gulp.dest(tsfile[0].replace(srcPath,distPath)))
        }))

        for(let notTsfilePath of notTsFilePaths){
            gulp.src(path.join(notTsfilePath[0],notTsfilePath[1]))
                .pipe(gulp.dest(distLibPath))    
        }

        for(let libName in libConfig){
            gulp.src(path.join(baseDir,libConfig[libName]))
                .pipe(rename(libName))
                .pipe(gulp.dest(distLibPath))
        }
        return true
    }catch(e){
        return false
    }
}

async function build_ts(){
    try{
        let paths = await pathsInPage()
        let tsfiles = paths.filter(_path => path.extname(_path[1]) === '.ts')
        // console.log(tsfiles)
    
        await Promise.all(tsfiles.map(async function(tsfile){
            var bundleStream = browserify(path.join(...tsfile)).plugin('tsify').ignore('../../lib/console').bundle()
            // var bundleStream = browserify(path.join(...tsfile).plugin('tsify').bundle()
            bundleStream
                .pipe(source(tsfile[1]))
                .pipe(rename(tsfile[1].replace('.ts','.js')))
                .pipe(gulp.dest(tsfile[0].replace(srcPath,distPath)))
        }))
        return true
    }catch(e){
        return false
    }
}

function readdir(targetPath){
    return new Promise(function(exec){
        fs.readdir(targetPath,(err,paths)=>{
            if(err){
                console.log("fs.readdir error")
                console.log(err)
            }
            exec(paths.map(path=>[targetPath,path]))
        })
    })
}

async function pathsInSrc(){
    try{
        let files = await readdir(srcPath)
        return files
    }catch(e){
        console.log(e)
    }
}

async function pathsInLab(){
    try{
        let paths = await pathsInSrc()
        let promises = paths.map(_path=>path.join(..._path)).filter(file=>fs.lstatSync(file).isDirectory()).map((dir)=>{return readdir(dir)})
        let results =  (await Promise.all(promises)).reduce((prev,next)=>[...prev,...next],[])
        return results
        // console.log(results)
    }catch(e){
        console.log(e)
    }
}

async function pathsInPage(){
    try{
        let paths = await pathsInLab()
        let promises = paths.map(_path=>path.join(..._path)).filter(file=>fs.lstatSync(file).isDirectory()).map((dir)=>{return readdir(dir)})
        let results =  (await Promise.all(promises)).reduce((prev,next)=>[...prev,...next],[])
        return results
    }catch(e){
        console.log(e)
    }
}