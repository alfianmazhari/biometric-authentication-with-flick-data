function [model] = Training_pureSVM(featureSet,newDataset_old,newNegDataset,classiferNum,penaltyList,hiddenSizesNum)

%take feature data needed (certain featureSet)
%trainPosData = trainPosData(:,featureSet);
%trainNegData = trainNegData(:,featureSet);
%trainData = [trainPosData;trainNegData];

%trainData_expandedHistogram = [];
%for dataCount = 1:size(trainData,1)
	%combined all histogram (expanded histogram)
%    trainData_expandedHistogram = [trainData_expandedHistogram;MixHistogram(trainData(dataCount,:))];
%end

%newDataset_old, newNegDataset
trainData = [newDataset_old; newNegDataset];
trainDataAns = [ones(size(newDataset_old,1),1);zeros(size(newNegDataset,1),1)];

[model,probability] = ClassifierToolsTraining(classiferNum,penaltyList,trainData,trainDataAns,hiddenSizesNum);

end

