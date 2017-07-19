var data = [];

var m = 1;
var b = 0;
var ymean = 0;
var rSquareValue = 0;
var MSE = 0;
var RMSE = 0;

var Ftest = 0;
var studentT = 0;
var peason = 0;

function setup() {
	createCanvas(800, 800);
	background(51);
}

function linearRegression(){
	var xsum = 0;
	var ysum = 0;

	for(var i = 0; i < data.length; i ++){
		xsum += data[i].x;
		ysum += data[i].y;
	}

	var xmean = xsum / data.length;
	ymean = ysum / data.length;

	var fractalUpper = 0;
	var fractalLower = 0;

	for(var i = 0; i < data.length; i ++){
		fractalUpper += (data[i].x - xmean)*(data[i].y - ymean);
		fractalLower += (data[i].x - xmean)*(data[i].x - xmean);
	}
	m = fractalUpper / fractalLower;
	b = ymean - m*xmean;
}

function rSquare(){
	rSquareValue = 0;
	var yPredictValueSquareSum = 0;
	var ymeanSquareSum = 0;
	for(var i = 0; i < data.length; i ++){
		var yPredictValue = m * data[i].x + b;
		yPredictValueSquareSum += (data[i].y - yPredictValue)*(data[i].y - yPredictValue);
		ymeanSquareSum += (data[i].y - ymean)*(data[i].y - ymean);
	}
	rSquareValue = 1 - (yPredictValueSquareSum / ymeanSquareSum);
	MSE = yPredictValueSquareSum / data.length;
	RMSE = sqrt(MSE);
}

function drawLine(){
	var x1 = 0;
	var y1 = m * x1 + b;
	var x2 = 1;
	var y2 = m * x2 + b;

	x1 = map(x1, 0, 1, 0, width);
	y1 = map(y1, 0, 1, height, 0);
	x2 = map(x2, 0, 1, 0, width);
	y2 = map(y2, 0, 1, height, 0);

	stroke(255, 0, 255);
	line(x1, y1, x2, y2);
}

function mousePressed() {
	var x = map(mouseX, 0, width, 0, 1);
	var y = map(mouseY, 0, height, 1, 0);

	var point = createVector(x, y);
	data.push(point);
  }

  function draw(){
  	background(51);

  	for(var i = 0; i < data.length; i ++){
  		var x = map(data[i].x, 0, 1, 0, width);
  		var y = map(data[i].y, 0, 1, height, 0);
  		fill(255);
  		stroke(255);
  		ellipse(x, y, 8, 8);
  	}

  	rSquare();
  	linearRegression();
  	drawLine();
  	fill(175);
  	text("Sample Size: "+data.length, width-300, 30);
  	text("Goodness of Fit - R2: "+nfc(rSquareValue,3), width-300, 50);
  	text("MSE-Mean Squared Error: "+nfc(MSE,3), width-300, 70);
  	text("RMSE-Root Mean Squared Error: "+nfc(RMSE,3), width-300, 90);
  }