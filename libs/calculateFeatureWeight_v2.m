function [ expandedFeaturesWeight ] = calculateFeatureWeight_v2(KofKmean,TrainPosData,TrainNegData,centroidExpandedHistogram,featureBinNum)
modelWeight = [];
expandedFeaturesWeight = [];

%for all cluster centroid 
for kIndex = 1:size(centroidExpandedHistogram,1)
	
	% get the cluster centroid (in histogram) of corresponding kIndex
    kCentroidHist = centroidExpandedHistogram(kIndex,:);
	
	%if the current kIndex below or equal to KofKmean (i.e. 5,10,15)
    if kIndex <= KofKmean
        kthPosData = TrainPosData;
        negData = TrainNegData;
    else
        kthPosData = TrainNegData;
        negData = TrainPosData;
    end
	
	%calculate the weight using MyFeatureScore
    [weight,posdis,negdis] = MyFeatureScore_v2(kCentroidHist,kthPosData,negData,featureBinNum);
    %
    
    %modelWeight
    modelWeight = [modelWeight;weight];
    aModeltempweight = [];
    %17個struct展開為2483個
    %所以weight原本為17,要分別對應成2483
	
    for index = 1:numel(featureBinNum)
        tempweight = [];
        tempweight(1:featureBinNum(index)) = weight(index);
        aModeltempweight = [aModeltempweight tempweight];
    end
    expandedFeaturesWeight = [expandedFeaturesWeight;aModeltempweight];
end

end

