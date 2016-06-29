import scipy.io
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import ExtraTreesClassifier
import itertools
#functions for calculating features for use in seizure detection algorithm

def allFeatures(data):
    output = []
    chans = len(data.ix[1,:])
    freq = np.fft.fftfreq(len(data['time']),1.0/len(data['time']))
    maxFreqs = []
    times = len(data.ix[:,1])-1 
    featureList = []
    delta1 = data.values[1:,:] - data.values[:times,:] # the 1st derivative
    glob1 = pd.DataFrame(delta1)
    freq1 = np.fft.fftfreq(len(glob1.ix[:,0]),1.0/len(glob1.ix[:,0]))
    maxFreqs1 = []
    delta2 = delta1[1:,:] - delta1[:times-1,:] # the 2nd derivative
    glob2 = pd.DataFrame(delta2)
    freq2 = np.fft.fftfreq(len(glob2.ix[:,0]),1.0/len(glob2.ix[:,0]))
    maxFreqs2 = []
# Channel Features
    for i in range(1,chans):
        output.append(data.ix[:,i].abs().max())
        featureList.append('chan%iMaxAmp'%(i-1))
        output.append(data.ix[:,i].abs().mean())
        featureList.append('chan%iMeanAmp'%(i-1))
        output.append(data.ix[:,i].abs().var()) 
        featureList.append('chan%iVarAbs'%(i-1))   
        output.append(data.ix[:,i].var())
        featureList.append('chan%iVar'%(i-1))
        output.append(abs(np.fft.fft(data.ix[:,i])).max())
        featureList.append('chan%iMaxFourAmp'%(i-1))
        output.append(abs(np.fft.fft(data.ix[:,i])).mean())
        featureList.append('chan%iMeanFourAmp'%(i-1))
        output.append(abs(np.fft.fft(data.ix[:,i])).var())
        featureList.append('chan%iVarFourAmp'%(i-1))
        ft = abs(np.fft.fft(data.ix[:,i]))
        output.append(abs(freq[np.argmax(ft)]))
        featureList.append('chan%iMaxFreq'%(i-1))
# 1st Derivative Channel Features
        output.append(np.max(np.abs(delta1[:,i])))
        featureList.append('chan%iMaxDel1'%(i-1))
        output.append(np.mean(np.abs(delta1[:,i])))
        featureList.append('chan%iMeanDel1'%(i-1))
        output.append(np.var(np.abs(delta1[:,i])))
        featureList.append('chan%iVarAbsDel1'%(i-1))
        output.append(np.var(delta1[:,i]))
        featureList.append('chan%iVarDel1'%(i-1))
        output.append(np.max(np.abs(np.fft.fft(delta1[:,i]))))
        featureList.append('chan%iMaxDel1Four'%(i-1))
        output.append(np.mean(np.abs(np.fft.fft(delta1[:,i]))))
        featureList.append('chan%iMeanDel1Four'%(i-1))
        output.append(np.var(np.abs(np.fft.fft(delta1[:,i]))))
        featureList.append('chan%iVarDel1Four'%(i-1))
        ft = np.abs(np.fft.fft(delta1[:,i]))
        output.append(abs(freq1[np.argmax(ft)]))
        featureList.append('chan%iMaxDel1Freq'%(i-1))
# 2nd Derivative Channel Features
        output.append(np.max(np.abs(delta2[:,i])))
        featureList.append('chan%iMaxDel2'%(i-1))
        output.append(np.mean(np.abs(delta2[:,i])))
        featureList.append('chan%iMeanDel2'%(i-1))
        output.append(np.var(np.abs(delta2[:,i])))
        featureList.append('chan%iVarAbsDel2'%(i-1))
        output.append(np.var(delta2[:,i]))
        featureList.append('chan%iVarDel2'%(i-1))
        output.append(np.max(np.abs(np.fft.fft(delta2[:,i]))))
        featureList.append('chan%iMaxDel2Four'%(i-1))
        output.append(np.mean(np.abs(np.fft.fft(delta2[:,i]))))
        featureList.append('chan%iMeanDel2Four'%(i-1))
        output.append(np.var(np.abs(np.fft.fft(delta2[:,i]))))
        featureList.append('chan%iVarDel2Four'%(i-1))
        ft = np.abs(np.fft.fft(delta2[:,i]))
        output.append(abs(freq2[np.argmax(ft)]))
        featureList.append('chan%iMaxDel2Freq'%(i-1))
# Global Features
    output.append(data.ix[:,1:].abs().apply(np.max).max())
    featureList.append('maxAmp')
    output.append(data.ix[:,1:].abs().apply(np.max).mean())
    featureList.append('meanAmp')
    output.append(data.ix[:,1:].abs().apply(np.max).var())
    featureList.append('varAmpAbs')
    output.append(data.ix[:,1:].apply(np.max).var())
    featureList.append('varAmp')
    output.append(data.ix[:,1:].abs().apply(np.mean).var())
    featureList.append('varMean')
    output.append(data.ix[:,1:].abs().apply(np.var).var())
    featureList.append('varVar')
    output.append(data.ix[:,1:].abs().apply(np.var).mean())
    featureList.append('meanVar')
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).max())
    featureList.append('maxFourAmp')
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).mean())
    featureList.append('meanFourAmp')
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).var())
    featureList.append('varFourAmp')
    for i in range(len(data.columns)-1):
        ft = abs(np.fft.fft(data['chan%i'%i]))
        maxFreqs.append(abs(freq[np.argmax(ft)]))
    output.append(np.max(maxFreqs))
    featureList.append('maxFreq')
    output.append(np.mean(maxFreqs))
    featureList.append('meanFreq')
    output.append(np.var(maxFreqs))
    featureList.append('varFreq')
    combs = []
    [combs.append(a) for a in (itertools.combinations(list(data.columns)[1:], 2))] 
    covs = [np.cov(data.ix[:,combs[k][0]], data.ix[:,combs[k][1]])[0,1] for k in np.arange(0,len(combs)) ]
    output.append(np.mean(np.abs(covs)))
    featureList.append('coVar')
# 1st Derivative Global Features
    output.append(glob1.ix[:,1:].abs().apply(np.max).max())
    featureList.append('maxDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.max).mean())
    featureList.append('meanDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.max).var())
    featureList.append('varDel1Abs')
    output.append(glob1.ix[:,1:].apply(np.max).var())
    featureList.append('varDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.mean).var())
    featureList.append('varMeanDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.var).var())
    featureList.append('varVarDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.var).mean())
    featureList.append('meanVarDel1')
    output.append(np.array([abs(np.fft.fft(glob1.ix[:,i])).max() for i in range(len(glob1.columns) - 1)]).max())
    featureList.append('maxFourDel1')
    output.append(np.array([abs(np.fft.fft(glob1.ix[:,i])).max() for i in range(len(glob1.columns) - 1)]).mean())
    featureList.append('meanFourDel1')
    output.append(np.array([abs(np.fft.fft(glob1.ix[:,i])).max() for i in range(len(glob1.columns) - 1)]).var())
    featureList.append('varFourDel1')
    for i in range(len(glob1.columns)-1):
        ft = abs(np.fft.fft(glob1.ix[:,i]))
        maxFreqs1.append(abs(freq1[np.argmax(ft)]))
    output.append(np.max(maxFreqs1))
    featureList.append('maxFreqDel1')
    output.append(np.mean(maxFreqs1))
    featureList.append('meanFreqDel1')
    output.append(np.var(maxFreqs1))
    featureList.append('varFreqDel1')
    combs = []
    [combs.append(a) for a in (itertools.combinations(list(glob1.columns)[1:], 2))] 
    covs = [np.cov(glob1.ix[:,combs[k][0]], glob1.ix[:,combs[k][1]])[0,1] for k in np.arange(0,len(combs)) ]
    output.append(np.mean(np.abs(covs)))
    featureList.append('coVarDel1')
# 2nd Derivative Global Features
    output.append(glob2.ix[:,1:].abs().apply(np.max).max())
    featureList.append('maxDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.max).mean())
    featureList.append('meanDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.max).var())
    featureList.append('varDelAbs2')
    output.append(glob2.ix[:,1:].apply(np.max).var())
    featureList.append('varDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.mean).var())
    featureList.append('varMeanDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.var).var())
    featureList.append('varVarDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.var).mean())
    featureList.append('meanVarDel2')
    output.append(np.array([abs(np.fft.fft(glob2.ix[:,i])).max() for i in range(len(glob2.columns) - 1)]).max())
    featureList.append('maxFourDel2')
    output.append(np.array([abs(np.fft.fft(glob2.ix[:,i])).max() for i in range(len(glob2.columns) - 1)]).mean())
    featureList.append('meanFourDel2')
    output.append(np.array([abs(np.fft.fft(glob2.ix[:,i])).max() for i in range(len(glob2.columns) - 1)]).var())
    featureList.append('varFourDel2')
    for i in range(len(glob2.columns)-1):
        ft = abs(np.fft.fft(glob2.ix[:,i]))
        maxFreqs2.append(abs(freq2[np.argmax(ft)]))
    output.append(np.max(maxFreqs2))
    featureList.append('maxFreqDel2')
    output.append(np.mean(maxFreqs2))
    featureList.append('meanFreqDel2')
    output.append(np.var(maxFreqs2))
    featureList.append('varFreqDel2')
    combs = []
    [combs.append(a) for a in (itertools.combinations(list(glob2.columns)[1:], 2))] 
    covs = [np.cov(glob2.ix[:,combs[k][0]], glob2.ix[:,combs[k][1]])[0,1] for k in np.arange(0,len(combs)) ]
    output.append(np.mean(np.abs(covs)))
    featureList.append('coVarDel2')
    return pd.DataFrame({'allFeats':output},index=featureList).T







def setDataDirectory(dirName):
    global dataDirectory
    dataDirectory = dirName
    return

def loadData(matlabFile,lat=0):
    matlabDict = scipy.io.loadmat(matlabFile)
    if matlabFile.count('_ictal_') > 0:
        lat = matlabDict['latency'][0]
    freq = len(matlabDict['data'][0])
    data = pd.DataFrame({'time':np.arange(lat,1.0+lat,1.0/freq)})
    for i in range(len(matlabDict['channels'][0][0])):
        channelName = "chan%i"%i
        data[channelName] = matlabDict['data'][i]
    return data

def downSample(data,factor):
    coarseData = data.groupby(lambda x: int(np.floor(x/factor))).mean()
    return coarseData
	
def convertToFeatureSeries(funcDict,data,featureFunctions,isSeizure=False,latency=0,isTest=False,testFile=""):
    #converts time series data into a set of features
    #featureFunctions should be a list of the desired features, which must be defined in funcDict
    #isSeizure and latency are used to add that information for training/validation
    #when loading test samples, isTest should be set True and the file name specified so that this information is available when writing the submission file
    #global funcDict
    data['time'] = data['time'] - latency
    features = []
    for func in featureFunctions:
        features.append(funcDict[func](data))
    data = pd.concat(features,axis=1)
    if not isTest:
        data['latency'] = latency
        data['isSeizure'] = int(isSeizure)
        data['isEarly'] = int(latency < 19 and isSeizure)
    else:
        data['testFile'] = testFile
    return data

def loadTrainAndValidationSamples(funcDict,dataDirectory,dataSelector,featureFunctions,commonFrequency=-1):
    
    #loads training samples and optionally splits off a chunk for validation
    #dataSelector is a list of lists that have the form [patientName,fraction of seizure segments to use for validation,fraction of non-seizure segments for validation
    #--for example [['Patient_2',0.5,0.5],['Dog_3',0.2,0.2]] would load the data for patient 2 and dog 3, putting half of the patient 2 data and 20% of the dog 3 data in the validation sample and the rest in the training sample
    #featureFunctions specifies the list of features to use
    #commonFrequency option is used to downsample the data to that frequency
    entriesTrain = []
    entriesValid = []
    for patient in dataSelector:
        files = os.listdir('%s/%s'%(dataDirectory,patient[0]))
        ictal = []
        interictal = []
        for phil in files:
            if phil.count('_inter') > 0:
                interictal.append(phil)
            elif phil.count('_ictal_') > 0:
                ictal.append(phil)
        for i in ictal:
            tmpData = loadData("%s/%s/%s"%(dataDirectory,patient[0],i))
            lat = tmpData['time'][0]
            if commonFrequency > 0:
                downFactor = float(len(tmpData['time'])) / commonFrequency
                if downFactor > 1.0:
                    tmpData = downSample(tmpData,downFactor)
            featureSet = convertToFeatureSeries(funcDict,tmpData,featureFunctions,True,lat)
            if np.random.random() > patient[1]:
                entriesTrain.append(featureSet)
            else:
                entriesValid.append(featureSet)
        for ii in interictal:
            tmpData = loadData("%s/%s/%s"%(dataDirectory,patient[0],ii))
            lat = tmpData['time'][0]
            if commonFrequency > 0:
                downFactor = float(len(tmpData['time'])) / commonFrequency
                if downFactor > 1.0:
                    tmpData = downSample(tmpData,downFactor)
            featureSet = convertToFeatureSeries(funcDict,tmpData,featureFunctions,False,0)
            if np.random.random() > patient[2]:
                entriesTrain.append(featureSet)
            else:
                entriesValid.append(featureSet)
    if len(entriesTrain) == 0:
        print "ERROR: No entries in training sample"
        return {'train':0,'validation':0}
    trainSample = pd.concat(entriesTrain,ignore_index=True)
    if len(entriesValid) == 0:
        return {'train':trainSample,'validation':0}
    validSample = pd.concat(entriesValid,ignore_index=True)
    return {'train':trainSample,'validation':validSample}
	#loads test data
    #arguments same as corresponding arguments for loadTrainAndValidationSamples
    patientList = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']
    entries = []
    for patient in patientList:
        files = os.listdir('%s/%s'%(dataDirectory,patient))
        for phil in files:
            if phil.count('test') > 0:
                tmpData = loadData("%s/%s/%s"%(dataDirectory,patient,phil))
                if commonFrequency > 0:
                    downFactor = float(len(tmpData['time'])) / commonFrequency
                    if downFactor > 1.0:
                        tmpData = downSample(tmpData,downFactor)
                featureSet = convertToFeatureSeries(funcDict,tmpData,featureFunctions,isTest=True,testFile=phil)
                entries.append(featureSet)
    testSample = pd.concat(entries,ignore_index=True)
    return testSample
    
def loadIndivTestSamples(funcDict,dataDirectory,dataSelector,featureFunctions,commonFrequency=-1):
    #loads test data
    #arguments same as corresponding arguments for loadTrainAndValidationSamples
    #patientList = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']
    entries = []
    for patient in dataSelector:
        files = os.listdir('%s/%s'%(dataDirectory,patient[0]))
        for phil in files:
            if phil.count('test') > 0:
                tmpData = loadData("%s/%s/%s"%(dataDirectory,patient[0],phil))
                if commonFrequency > 0:
                    downFactor = float(len(tmpData['time'])) / commonFrequency
                    if downFactor > 1.0:
                        tmpData = downSample(tmpData,downFactor)
                featureSet = convertToFeatureSeries(funcDict,tmpData,featureFunctions,isTest=True,testFile=phil)
                entries.append(featureSet)
    testSample = pd.concat(entries,ignore_index=True)
    return testSample
	#trains a random forest on the training sample and returns the trained forest
    #trainArray = trainDF.values
    #forest = RandomForestClassifier(n_estimators=1000)
    #return forest.fit(trainArray[:,0:-3],trainArray[:,-2:])
    #prints efficiency and false positive metrics and plots efficiency vs. latency for a given forest using the validation sample
    #output = forest.predict(validDF.values[:,0:-3])
    #validDF['PiS'] = output[:,0].astype(int)
    #validDF['PiE'] = output[:,1].astype(int)
    #return validDF

def trainDoubleForest(trainDF):
    #trains a random forest on the training sample and returns the trained forest
    trainArray = trainDF.values
    forestSeizure = ExtraTreesClassifier(n_estimators=1000, min_samples_split = 1)
    forestEarly = ExtraTreesClassifier(n_estimators=1000, min_samples_split = 1)
    return {'seizure':forestSeizure.fit(trainArray[:,0:-3],trainArray[:,-2]),'early':forestEarly.fit(trainArray[:,0:-3],trainArray[:,-1])}
   
   

def testProbs(forestList,testDF):
    #runs the forest on the test sample and returns output
    output = []
    for forest in forestList:
        output.append(forest.predict_proba(testDF.values[:,0:-1])[:,1])
    output = np.array(output).T
    return output 
    
#select all dogs/patients, don't set aside any data for validation

    dataDirectory = "C:\Users\Abhinav\Desktop\s\data"  
    #Dictionary associates functions with string names so that features can be easily selected later
    funcDict = {'allFeats'      : allFeatures}
    
    dataSelector = [['Dog_1',0,0]]
                    
    predictions = []; testSamples = pd.DataFrame()
                    
    for num, dataSet in enumerate(dataSelector):
        #print dataSet, num
        #print "Loading train/validation samples using selector:\n",dataSelector[num]
        samples = loadTrainAndValidationSamples([dataSet],['allFeats'],100.0)
        #print "Training sample size: ",samples['train'].shape
        forest = trainDoubleForest(samples['train'])
        
       
        
        #print "Done training. Loading test samples..."
        testSam = loadIndivTestSamples([dataSet], ['allFeats'],100.0) 
        testSamples = pd.concat([testSamples, testSam])    
        #print "Test sample size: ",testSam.shape
        predictions.extend(testProbs([forest['seizure'],forest['early']],testSam )) 
        
        frame=pd.DataFrame(np.array(predictions),columns=['seizure','early'])
        print frame
        
   
makeSubmit(np.array(predictions), testSamples)

print "Done."
