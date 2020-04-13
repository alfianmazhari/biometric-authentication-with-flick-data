function [distanceBasedFeatureData] = kernelMapping(centroidWithWeight,expandedHistogram, expFeaturesWeight )

%complete kernel map, with featureweight

distanceBasedFeatureData = [];
numOfData = size(expandedHistogram, 1);

%for each model
for modelIndex = 1:size(centroidWithWeight,1)
	%replicate the value centroidWithWeight
    theDatatrainModel = repmat(centroidWithWeight(modelIndex,:),size(expandedHistogram, 1),1);
	%replicate the value of weightforExpData
    weightforExpData = repmat(expFeaturesWeight(modelIndex,:),size(expandedHistogram, 1),1);
	%calcute expDataWithWeight
    expDataWithWeight = weightforExpData .* expandedHistogram;
    %∫‚DataªP∏s§§§ﬂ∫‚Æt≤ß
	%calcute the distance using KLDivergence
    dataTrainFeature = KLDivergence(theDatatrainModel',expDataWithWeight');
	%save it
    distanceBasedFeatureData = [distanceBasedFeatureData dataTrainFeature'];
end
end