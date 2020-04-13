function [probability] = Testing_pureSVM_transferLearning(featureSet,testPosData,testNegData,model,classiferNum, transformation)
   
%TESTINGMODEL Summary of this function goes here
%   Detailed explanation goes here

testPosData = testPosData(:,featureSet);
testNegData = testNegData(:,featureSet);
testDataAns = [ones(size(testPosData,1),1);zeros(size(testNegData,1),1)];

%expand pos data
testPosData_expandedHistogram = [];
for dataCount = 1:size(testPosData, 1)
    testPosData_expandedHistogram = [testPosData_expandedHistogram; MixHistogram(testPosData(dataCount,:))];
end

%get bin width, just look at first data, assumed the rest of data will be
%the same
firstData = testPosData(1,:);
binWidth = [];
for featureCount = 1:size(firstData,2);
    binWidth = [binWidth; size(firstData(1,featureCount).histogram,2)];
end

%transform pos data, then normalize
transformed_testPosData_expandedHistogram = testPosData_expandedHistogram + (repmat(transformation,size(testPosData_expandedHistogram,1),1));
transformed_testPosData_expandedHistogram(transformed_testPosData_expandedHistogram < 0) = 0;
transformed_testPosData_expandedHistogram(transformed_testPosData_expandedHistogram > 1) = 1;

%normalization part
normalized_transformed_testPosData = [];
for transformedDataCount = 1:size(transformed_testPosData_expandedHistogram, 1)
    normedHist = [];
    binIndex = 1;
    for featureCount = 1:size(binWidth, 1)
        minBin = binIndex;
        maxBin = binIndex + binWidth(featureCount) -1; 
    
        %normalize here
        singleHist = transformed_testPosData_expandedHistogram(transformedDataCount,minBin:maxBin);
        normedSingleHist = singleHist ./ sum(singleHist(:));
        normedHist = [normedHist normedSingleHist];
        
        binIndex = maxBin + 1;
    end
    normalized_transformed_testPosData = [normalized_transformed_testPosData; normedHist];
end

testNegData_expandedHistogram = [];
for dataCount = 1:size(testNegData, 1)
    testNegData_expandedHistogram = [testNegData_expandedHistogram; MixHistogram(testNegData(dataCount,:))];
end

%testData_expandedHistogram = [normalized_transformed_testPosData;testNegData_expandedHistogram];
testData_expandedHistogram = [transformed_testPosData_expandedHistogram;testNegData_expandedHistogram];

probability = ClassifierToolsTest(classiferNum, model, testDataAns, testData_expandedHistogram);

end
