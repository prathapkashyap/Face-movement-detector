

var express=require ('express');
const controller=require('./controller')
// var controller=require('./controller/frontcontroller.js')
// var path=require('path')
// var body=require('body-parser');
//start the express function
var app=express();
//set the view engine as ejs so that data can be sent from this page to other .ejs files
// app.use(body.urlencoded({extended:false}));
app.set('view engine','ejs');
//middleware to get the static files like css working

app.use('/public',express.static('public'));

//fire controller
controller(app);

//start the server
app.listen(4000);
console.log('its working');
