from utils import *

#select all dogs/patients, don't set aside any data for validation
dataSelector = [['Dog_1',0,0]]
                
predictions = []; testSamples = pd.DataFrame()
                
for num, dataSet in enumerate(dataSelector):
    print dataSet, num
    print "Loading train/validation samples using selector:\n",dataSelector[num]
    samples = loadTrainAndValidationSamples([dataSet],['allFeats'],100.0)
    print "Training sample size: ",samples['train'].shape
    forest = trainDoubleForest(samples['train'])
    print "Done training. Loading test samples..."

    testSam = loadIndivTestSamples([dataSet], ['allFeats'],100.0) 
    testSamples = pd.concat([testSamples, testSam])    
    print "Test sample size: ",testSam.shape
    predictions.extend(testProbs([forest['seizure'],forest['early']],testSam )) 
   
makeSubmit(np.array(predictions), testSamples)

print "Done."

