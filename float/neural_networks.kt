import java.io.File
import kotlin.math.*
import kotlin.random.Random
import java.io.FileInputStream
//Andrew Sarangan

class Network(I:Int, layers:List<Int>, Rmax:Double, val function:String) {
    //I is the number of inputs. It just determines the number of columns of the first matrix.  
    //layers is a list of numbers specifying how many rows in the intermediate layers. Last item in list is the number of outputs.
    //Listed items are from left to right.
    //Rmax is the max value of the random number (from negative to positive)
    val b:Array<Array<Float>> = layers.map { Array(it) {Random.nextDouble(-Rmax,Rmax).toFloat()} }.toTypedArray()
    val w:Array<Array<Array<Float>>> = (listOf(I)+layers).zipWithNext().map { (y1, y2) -> Array(y2) 
                                                { Array(y1) {Random.nextDouble(-Rmax,Rmax).toFloat()} } }.toTypedArray()
    val gradB:Array<Array<Float>> = layers.map { Array(it) {0.0f} }.toTypedArray()
    val gradW:Array<Array<Array<Float>>> = (listOf(I)+layers).zipWithNext().map { (y1, y2) -> Array(y2) 
                                                { Array(y1) {0.0f} } }.toTypedArray()
	val a:Array<Array<Float>> = layers.map { Array(it) {0.0f} }.toTypedArray()
	val d:Array<Array<Float>> = layers.map { Array(it) {0.0f} }.toTypedArray()
    val z:Array<Array<Float>> = layers.map { Array(it) {0.0f} }.toTypedArray()
    
    //Set the activation function as the extension to the Float class. We can add other functions here. Default is the sigmoid. Input is z
    fun Array<Float>.act(type:String):Array<Float> {	val s = this.map{exp(it)}.sum()
    													return map{
        													when (type) { 	
        														"sigmoid" -> 1.0f/(1.0f+exp(-it))
    															"tanh"	-> tanh(it)
                                                    			"relu"	-> max(0.0f,it)
                                                        		"softmax" -> exp(it)/s
        													else -> 1.0f/(1.0f+exp(-it))}}.toTypedArray()}
    
    //Derivative of the activation function, also set as an extension to the Float class, returning a matrix because of the cross terms. 
    //Default is the sigmoid. Input is z
    fun Array<Float>.dact(type:String):Array<Array<Float>>{
        											val a = this.act(type)
        											val dadz = Array(this.size){Array(this.size){0.0f}}  
        											when (type) {	
                                                        "sigmoid" -> 	dadz.forEachIndexed{i,_ -> dadz[i][i] = a[i]*(1.0f-a[i])}
                                                    	"tanh"	-> dadz.forEachIndexed{i,_ -> dadz[i][i] = 1.0f - a[i].pow(2)}
                                                    	"relu"	-> dadz.forEachIndexed{i,_ -> if (a[i] > 0.0f){dadz[i][i] = 1.0f} else {dadz[i][i] = 0.0f}}
                                                        "softmax" -> dadz.forEachIndexed{i,v -> v.forEachIndexed{j,_ -> if (i==j) {v[j] = a[i]*(1.0f-a[i])}
                                                        																		else{v[j] = -a[j]*a[i]}}}
        												else -> dadz.forEachIndexed{i,_ -> dadz[i][i] = a[i]*(1.0f-a[i])}}
                                                    return dadz}
    	
    //Step forward through all layers, saving all the intermediate z and a-vectors
    fun forward(x:Array<Float>){	
        for (l in 0..a.size-1) {
                a[l].forEachIndexed{i,_ ->
                        z[l][i] = w[l][i].zip( if (l==0) x else a[l-1] ){j,k -> j*k}.sum()+b[l][i]}
                a[l] = z[l].act(function)}
    }
    
    //Transpose a matrix. It is declared as an extension to the Array<Array<Double>> class.
    fun Array<Array<Float>>.transpose():Array<Array<Float>> {
        val rows = this.size
        val cols = this.first().size
        return Array(cols){c ->
            Array(rows){r ->
                this[r][c]}}
    }
    
    //Quadratic cost function (only the last a-column is needed)
    fun cost(y:Array<Float>):Float{
        return a.last().mapIndexed{i,v -> (v-y[i]).pow(2)}.sum()
    }
    
    //Calculate the output accuracy (only the last a-column is needed)
    //The largest element of the output vector is treated as 1 and all the others as 0. 
    fun eval(y:Array<Float>):Boolean{
		var maxAIndex = 0
        a.last().forEachIndexed{i,v -> if (v > a.last()[maxAIndex]) {maxAIndex = i}}
        var maxYIndex = 0
        y.forEachIndexed{i,v -> if (v > y[maxYIndex]) {maxYIndex = i}}
        if (maxAIndex == maxYIndex) return true else return false
    }
    
    //Calculate all the delta values, moving backwards through the network
    fun deltas(y:Array<Float>){
        
        //Calculate the last layer delta
        val L = a.size-1
        var dadz = z[L].dact(function)
        z[L].forEachIndexed{i,z ->
            d[L][i] = (a[L][i]-y[i]) * dadz[i][i]}	//derivative of the cost function is here
        
        //Then calculate the other deltas moving backwards. 
        for (l in a.size-2 downTo 0){
            val wlplus1T = w[l+1].transpose()
            dadz = z[l].dact(function)
            z[l].forEachIndexed{i,z ->
                d[l][i] = wlplus1T[i].zip(d[l+1]){j,k -> j*k}.sum() * dadz[i][i] }}
    }
    
    //Calculate the gradients of the w matrix elements. This keeps cumulating until we reset it.
    fun gradient(x:Array<Float>, y:Array<Float>){
		forward(x)
		deltas(y)
        w.forEachIndexed{l,wl ->
            wl.forEachIndexed{r,y -> 
                gradB[l][r] += d[l][r]
                y.forEachIndexed{c,_ ->
                    if (l == 0){ gradW[l][r][c] += d[l][r]*x[c] } else { gradW[l][r][c] += d[l][r]*a[l-1][c] }}}}
    }
    
    //After calculating the gradients of one batch, call this function to reset the gradients to zero.
	fun gradientReset(){
        w.forEachIndexed{l,wl ->
            wl.forEachIndexed{r,y ->
                gradB[l][r] = 0.0f
                y.forEachIndexed{c,_ ->
                    gradW[l][r][c] = 0.0f}}}
    }
        
    //Apply the gradients to improve w and b
    fun improveWB(lrate:Float){
        gradW.forEachIndexed{l,wl ->
            wl.forEachIndexed{r,y ->
                b[l][r] -= gradB[l][r]*lrate
                y.forEachIndexed{c,_ ->
                    w[l][r][c] -= gradW[l][r][c]*lrate}}}}
}	//End of Network class

//This class will for reading MNIST files
class MNIST(imageFile:String, labelFile:String, count:Int){
    val yc:List<Array<Float>>
    val xc:List<Array<Float>>
    init{
        //We will read the whole file and then only take the amount of data requested
    	print("Reading y labels...")
        var bytes = FileInputStream(File(labelFile)).readBytes().drop(8) //There are 8 extra leading bytes in the file
        var strip = if(count==-1){0} else{bytes.size - count}
        bytes = bytes.drop(strip)
    	val y = MutableList<Float>((bytes.size)*10){0.0f}
    	bytes.forEachIndexed{i,v -> y[v+i*10] = 1.0f}
		yc = y.chunked(10).map{v -> v.toTypedArray()}
    	println("Finished Reading y")
    
    	print("Reading x images...")
        bytes = FileInputStream(File(imageFile)).readBytes().drop(16) //There are 16 extra leading bytes in the file
        strip = if(count==-1){0} else{bytes.size - count*784}
        bytes = bytes.drop(strip)
    	xc = bytes.toList().chunked(784).map{v ->
         				v.map{w -> (w.toUByte().toFloat()/255.0f)}.toTypedArray()}.toMutableList()
    	println("Finished Reading x")}

    fun showByte(byteNum:Int){
        var pgm = "P2\n28 28\n255\n"
        yc[byteNum].forEach{v -> val str = "%02X ".format(v.toInt()).takeLast(3); print (str)}; println()   
        xc[byteNum].forEachIndexed{i,v -> 
            val str = "%02X ".format((v*255.0).toInt()).takeLast(3)
            print (str)
            pgm += (v*255.0).toInt().toString()+" "
            if ((i+1)%28==0){println(); pgm += "\n"}}
        println()
        File("file.pgm").writeText(pgm)
    }
}

//This is a sinusoidal test data on an input grid of 100. The output vector has 10 elements, which has
//the number of oscillation peaks in the input grid. 
//A random number can be added to the sine during evaluation to increase noise. Rmax is the magnitude.
class testData(Rmax:Double){
    val x:List<Array<Float>>
    val y:List<Array<Float>>
    init{
        x = (1..10).map{p -> Array<Float>(100){i -> sin((i*p*PI/100.0).toFloat()).pow(2)+Random.nextDouble(0.0,Rmax).toFloat()}}
        y = (1..10).map{p -> Array<Float>(10){i -> if (i+1==p) 1.0f else 0.0f}}
    }
}

//Parse the input arguments into keys and tokens
fun parseArguments(args: Array<String>): Map<String, String> {
    val keyTokens = mutableMapOf<String, String>()
    if (args[0].contains("=")){
        for (arg in args) {
            val parts = arg.split("=", limit = 2)
            val key = parts[0]
            val token = parts[1]
            keyTokens[key] = token}}
    return keyTokens
}

fun printEval(xyData:List<Pair<Array<Float>,Array<Float>>>, n:Network, epoch:Int):Float{
    var corrects = 0
    xyData.forEach{(x,y) -> 
        n.forward(x)
        corrects += if (n.eval(y)) 1 else 0}
    val correctFrac = corrects.toFloat()/xyData.size.toFloat()
    println("Epoch = %d\tAccuracy: %d/%d = %.1f%%".format(epoch,corrects,xyData.size,correctFrac*100.0))
    return correctFrac} 


fun main(args: Array<String>) {  
    val keyTokens = parseArguments(args)
    val dataSource = keyTokens["dataSource"]; println("dataSource = $dataSource")
    val trainDataN = keyTokens["trainDataN"]?.toInt()?:-1; println("trainDataN = $trainDataN")
    val function = keyTokens["function"]?:"sigmoid"; println("function = $function")
    val batchSize = keyTokens["batchSize"]?.toInt()?:1; println("batchSize = $batchSize")
    var LR = keyTokens["LR"]?.toFloat()?:1.0f; println("LR = $LR")
    val M = keyTokens["M"]?.toInt()?:30; println("M = $M")
    val MRMax = keyTokens["MRMax"]?.toDouble()?:1.0; println("MRMax = $MRMax")
    val trainingThreshold = keyTokens["trainingThreshold"]?.toFloat()?:0.95f; println("trainingThreshold = $trainingThreshold")
    val testDataN = keyTokens["testDataN"]?.toInt()?:-1; println("testDataN = $testDataN")
    val TRMax = keyTokens["TRMax"]?.toDouble()?:1.0; println("TRMax = $TRMax")

    val trainingData:List<Pair<Array<Float>,Array<Float>>>
    when (dataSource) {
        "MNIST" -> {
    		val MNISTTrainData = MNIST("../data/train-images.idx3-ubyte","../data/train-labels.idx1-ubyte", trainDataN)
    		trainingData = MNISTTrainData.xc.zip(MNISTTrainData.yc)
    		MNISTTrainData.showByte(1)
    		}
        else -> {
            val sineTrainData = testData(1.0e-6)
    		trainingData = sineTrainData.x.zip(sineTrainData.y)}
    }

    
    val (x,y) = trainingData[0]
    val I = x.size						//Number of inputs
    val O = y.size						//Number of outputs
    println("Number of Inputs = $I")
    println("Number of Outputs = $O")
    
    val n = Network(I, listOf(M,O), MRMax, function)
    val miniBatches = trainingData.chunked(batchSize)
   
   //Learning
   var epoch = 0
   var correctFrac:Float
   correctFrac = printEval(trainingData,n,epoch)
   while (correctFrac < trainingThreshold){    
        miniBatches.forEach{
        	it.forEach{(x,y) -> n.gradient(x,y)}
            n.improveWB(LR)
        	n.gradientReset()}
        epoch++
        correctFrac = printEval(trainingData,n,epoch)
   }    
    
    //Testing
    val testData:List<Pair<Array<Float>,Array<Float>>>
    when (dataSource) {
        "MNIST" -> {
            val MNISTTestData = MNIST("../data/t10k-images.idx3-ubyte","../data/t10k-labels.idx1-ubyte",testDataN)
            testData = MNISTTestData.xc.zip(MNISTTestData.yc)
    		//MNISTTestData.showByte(1)
        	}
        else -> {
    		val sineTestData = testData(TRMax)
    		testData = sineTestData.x.zip(sineTestData.y)}
    }
    println("Test Data:")
    printEval(testData,n,epoch)
}
