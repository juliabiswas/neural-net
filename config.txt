overall structure of user-specifications file:
train or test 
inputNodes hiddenLayers hiddenNodes (list all of them) outputNodes
inputActivations expectedOutputs (only for test)
iterations (only for train)
lambda (only for train)
lambdaChange (only for train)
lambdaMin lambdaMax (only for train)
desiredError (only for train)
random or specified (only for train)
minRandomWeight maxRandomWeight OR (if specified or test) weights
adaptive or notAdaptive (only if training)
testCases (only for train)
"file" OR "infile" OR "pelsFile"
inputActivations expectedOutputs (for each test case) (only for train)

(1) training with random weights
"train"
inputNodes hiddenLayers hiddenNodes (list all of them) outputNodes
iterations
lambda
lambdaChange
lambdaMin lambdaMax
desiredError
"random"
minRandomWeight maxRandomWeight
adaptive or notAdaptive
testCases
"file" OR "infile" OR "pelsFile"
file name for input activations and expected outputs (if "file")
file name for input activations (same as expected outputs) (if "pelsfile")
inputActivations expectedOutputs (for each test case)

(2) training with specified weights
"train"
inputNodes hiddenLayers hiddenNodes (list all of them) outputNodes
iterations
lambda
lambdaChange
lambdaMin lambdaMax
desiredError
"specified"
weights (list them)
adaptive or notAdaptive
testCases
"file" OR "infile" OR "pelsFile"
file name for input activations and expected outputs (if "file")
file name for input activations (same as expected outputs) (if "pelsfile")
inputActivations expectedOutputs (for each test case)

(3) testing
"test"
inputNodes hiddenLayers hiddenNodes (list all of them) outputNodes
weights (list them)
"file" OR "infile" OR "pelsFile"
file name for input activations and expected outputs (if "file")
file name for input activations (same as expected outputs) (if "pelsfile")
inputActivations expectedOutputs (for each test case)
