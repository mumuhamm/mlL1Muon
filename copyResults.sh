mkdir results/$1

mv -v lutNN2/lutNN_omtfRegression.txt results/$1
mv -v lutNN2/lutNN_omtfClassifier.txt results/$1

mv -v pictures/canvasLutNN_canvasCostHist.png results/$1/

mv -v pictures/lutNN2.root results/$1/
mv -v pictures/omtfAnalyzerlutNN*.png results/$1/
mv -v pictures/omtfAnalyzer.root results/$1/
mv -v pictures/canvasLutNN_GradientTrain_*0000.png results/$1/
mv -v pictures/canvasLutNN_GradientTrain_1000.png results/$1/
mv -v pictures/canvasLutNN_GradientTrain_100.png results/$1/
mv -v pictures/canvasLutNN_GradientTrain_0.png results/$1/

ls -l results/$1/


