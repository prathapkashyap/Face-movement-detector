module.exports=function(app){
    app.get('/',function(req,res){
        res.render('index')
    })




    app.get('/project',function(req,res){
        res.render('project')
        const { spawn } = require('child_process');
    const pyProg = spawn('python', ['./auth.py']);

    // pyProg.stdout.on('data', function(data) {

    //     console.log(data.toString());
    //     res.write(data);
         res.end('end');
       // res.render('project')
    });

    app.get('/final',function(req,res){
        res.render('final')
        const { spawn } = require('child_process');
    const pyProg = spawn('python', ['./final.py']);
    res.end('end');

    })
        

    // })



}