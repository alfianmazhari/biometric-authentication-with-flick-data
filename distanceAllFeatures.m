clear;

userInvolved = [2:5 7:12 14:30 32:57 59:70];
posture = 'sit_long';

%parameter setting
numOfFlick = 5;
train_start = 11;
train_stop = 20;
test_start = 21;
test_stop = 30;

sampleset_path = ['.\1003_' num2str(numOfFlick) 'flicks_train' num2str(train_start) '-' num2str(train_stop) '_test' num2str(test_start) '-' num2str(test_stop)];
FeatureIndex = [1:49];

result = [];
for userCount = 1:numel(userInvolved)
    userID = userInvolved(userCount);
    load([sampleset_path '\' posture '\user_' num2str(userID) '_train_sampleSet.mat'], 'trainHistogram');
    load([sampleset_path '\' posture '\user_' num2str(userID) '_test_sampleSet.mat'], 'testHistogram');
    
    result_user = [];
    for featureCount = 1:numel(FeatureIndex)
        trainHistogramDetail = [];
        for i = 1:size(trainHistogram, 1)
            trainHistogramDetail = [trainHistogramDetail; trainHistogram(i,FeatureIndex(featureCount)).histogram];
        end
        
                
        testHistogramDetail = [];
        for i = 1:size(trainHistogram, 1)
            testHistogramDetail = [testHistogramDetail; testHistogram(i,FeatureIndex(featureCount)).histogram];
        end
          
        %euclidean distance
        %euclidDistance = sqrt(sum((trainHistogramDetail' - testHistogramDetail') .^ 2));
        %euclidDistance_mean = mean(euclidDistance);
    
        %KL Divergence
        %KLDistance = KLDivergence(trainHistogramDetail', testHistogramDetail');
        %KLDistance_mean = mean(KLDistance);
   
        %EMD
        bin = [1:size(trainHistogramDetail,2)];
        EMDistance = [];
        for rowCount = 1:size(trainHistogramDetail,1)
            [~,EMDistanceRow] = emd(bin', bin', trainHistogramDetail(rowCount,:)', testHistogramDetail(rowCount,:)', @gdf);
            EMDistance = [EMDistance; EMDistanceRow];
        end
        EMDistance_mean = mean(EMDistance);
    
        result_feature = EMDistance_mean;
        result_user = [result_user result_feature];
    end
result = [result; result_user];    
end
