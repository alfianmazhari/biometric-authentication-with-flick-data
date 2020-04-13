function [probability] = Testing_pureSVM_transferLearning_newFeatureRepresentation(featureSet,testPosData,...
    testNegData,model,classiferNum,centroidsWithWeight,featureWeight,combinedDataset,expandedNegHistogram,...
    trainNegDataCentroid,transformation,trainHistogram_old)
   
%TESTINGMODEL Summary of this function goes here
%   Detailed explanation goes here

testPosData = testPosData(:,featureSet);
testNegData = testNegData(:,featureSet);

trainHistogram_old = trainHistogram_old(:,featureSet);

testPosData_expandedHistogram = [];
for dataCount = 1:size(testPosData, 1)
    testPosData_expandedHistogram = [testPosData_expandedHistogram; MixHistogram(testPosData(dataCount,:))];
end

testNegData_expandedHistogram = [];
for dataCount = 1:size(testNegData, 1)
    testNegData_expandedHistogram = [testNegData_expandedHistogram; MixHistogram(testNegData(dataCount,:))];
end

combinedDataset_expandedHistogram = [];
for dataCount = 1:size(combinedDataset)
    combinedDataset_expandedHistogram = [combinedDataset_expandedHistogram; MixHistogram(combinedDataset(dataCount,:))];
end

trainHistogramOld_expandedHistogram = [];
for dataCount = 1:size(trainHistogram_old)
    trainHistogramOld_expandedHistogram = [trainHistogramOld_expandedHistogram; MixHistogram(trainHistogram_old(dataCount,:))]; 
end

testData = [testPosData_expandedHistogram; testNegData_expandedHistogram];

deltaMean = mean(trainHistogramOld_expandedHistogram) - mean(combinedDataset_expandedHistogram);

%ngitung centroid data gabungan
% pos_K_value = 10;
% neg_K_value = 20;
K_value = floor(size(trainHistogram_old,1)/2);
[clusterID, centroid, ~, distance] = kmeans(combinedDataset_expandedHistogram, K_value);
[combinedDataCentroids] = findClusterCentroid_v2(K_value, distance, clusterID, combinedDataset_expandedHistogram);
combinedDataCentroids = [combinedDataCentroids;trainNegDataCentroid];


featureWeight_combinedDataset = calculateFeatureWeight_v2(K_value,combinedDataset_expandedHistogram,...
    expandedNegHistogram,combinedDataCentroids,size(combinedDataset_expandedHistogram,2));
featureWeight_combinedDataset(featureWeight_combinedDataset == 1) = 0.6;
featureWeight_combinedDataset(featureWeight_combinedDataset == 0) = 0.4;

combinedDataCentroidsWithWeight = combinedDataCentroids .* featureWeight_combinedDataset;


% shift combined centroids
%nData = ceil(size(combinedDataCentroidsWithWeight,1)/2); 

%combinedDataCentroidsWithWeight = combinedDataCentroidsWithWeight - (repmat(deltaMean,size(combinedDataCentroidsWithWeight,1),1));
%combinedDataCentroidsWithWeight(combinedDataCentroidsWithWeight < 0) = 0;
%combinedDataCentroidsWithWeight(combinedDataCentroidsWithWeight > 1) = 1;

%newTestData = kernelMapping_v3(combinedDataCentroidsWithWeight,testData,featureWeight_combinedDataset);
% --------------------------------------------------------------

% shift old centroids
shifted_centroidsWithWeight = centroidsWithWeight - repmat(deltaMean,size(centroidsWithWeight,1),1);
shifted_centroidsWithWeight(shifted_centroidsWithWeight < 0) = 0;
shifted_centroidsWithWeight(shifted_centroidsWithWeight > 1) = 1;

newTestData = kernelMapping_v3(shifted_centroidsWithWeight,testData,featureWeight);
% --------------------------------------------------------------

%newTestData = kernelMapping_v3(centroidsWithWeight,testData,featureWeight);

%newTestData = kernelMapping_v3(combinedDataCentroidsWithWeight,testData,featureWeight);
%newTestData = kernelMapping_v3(combinedDataCentroidsWithWeight,testData,featureWeight_combinedDataset);

%newTestData = normalize(newTestData);

% transformed_newTestData = newTestData - (repmat(transformation,size(newTestData,1),1));
% transformed_newTestData(transformed_newTestData < 0) = 0;
% transformed_newTestData(transformed_newTestData > 1) = 1;

% n_transformedNewTestData = ceil(size(transformed_newTestData,1)/2);
% newTransformedNewTestPosData = transformed_newTestData(1:n_transformedNewTestData,:);
% newTransformedNewTestNegData = transformed_newTestData(n_transformedNewTestData+1:end,:);

%look for in which column the biggest value is
% nColumn = ceil(size(newTestData,2)/2);

% for i = 1:size(newTestData,1)    
%     minimum = min(min(newTestData(i,:)));
%     [x,y] = find(newTestData(i,:) == minimum);
%     if y <= nColumn
%         newTestData(i,:) - transformation;
%     end
% end



n_newTestData = ceil(size(newTestData,1)/2);
newTestPosData = newTestData(1:n_newTestData,:);
newTestNegData = newTestData(n_newTestData+1:end,:);

%get bin width, just look at first data, assumed the rest of data will be
%the same
% firstData = testPosData(1,:);
% binWidth = [];
% for featureCount = 1:size(firstData,2);
%     binWidth = [binWidth; size(firstData(1,featureCount).histogram,2)];
% end
% 
% %normalization part
% normalized_transformed_newPosTestPosData = [];
% for transformedDataCount = 1:size(transformed_newTestPosData, 1)
%     normedHist = [];
%     binIndex = 1;
%     for featureCount = 1:size(binWidth, 1)
%         minBin = binIndex;
%         maxBin = binIndex + binWidth(featureCount) -1; 
%     
%         %normalize here
%         singleHist = transformed_testPosData_expandedHistogram(transformedDataCount,minBin:maxBin);
%         normedSingleHist = singleHist ./ sum(singleHist(:));
%         normedHist = [normedHist normedSingleHist];
%         
%         binIndex = maxBin + 1;
%     end
%     normalized_transformed_testPosData = [normalized_transformed_testPosData; normedHist];
% end

%nggawe dataset anyar kanggo data negatif kanggo testing
%--------------------------------------------------------------

testDataAns = [ones(size(newTestPosData,1),1);zeros(size(newTestNegData,1),1)];

%testData_expandedHistogram = [transformed_newTestPosData;newTestNegData];

probability = ClassifierToolsTest(classiferNum, model, testDataAns, newTestData);

end
