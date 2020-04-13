clear
clc

addpath('.\TeachingMaterial');

numOfFlick = 5;
userInvolved = [1:4 7:12 14:30 35 36 39:57 59:70];
round = 1;
periods = [2 3 4 5 6 7 8];
posture = 'sit_long';

sampleset_path = ['.\1017_' num2str(numOfFlick) 'flicks_train_test_allperiod_v5'];    

for trainingCount = 1:1
        for testingCount = 1:7
            for userCount = 1:numel(userInvolved)

                userID = userInvolved(userCount);
%                 load([sampleset_path '\' posture '\user_' num2str(userID) '_period_2_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
%                 period2Data = trainHistogram;
%                 load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periods(trainingCount)) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
%                 trainHistogram_old = [period2Data; trainHistogram];

%                 load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periods(trainingCount)) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
%                 trainHistogram_old = trainHistogram;
                load([sampleset_path '\' posture '\user_' num2str(userID) '_period_2_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                trainHistogram_old = trainHistogram;
                
%                 load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periods(testingCount)) '_round_1_train_sampleSet.mat'], 'trainHistogram');
%                 trainHistogram_new = trainHistogram;
                
                if periods(trainingCount) == periods(testingCount) 
                    %load([sampleset_path '\' posture '\user_' num2str(userID) '_period_2_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                    %period2Data = trainHistogram;
                    load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periods(testingCount)) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                    %trainHistogram_new = trainHistogram;
                    %posDataTemp(userID).trainHistogram_new = [period2Data; trainHistogram];
                    posDataTemp(userID).trainHistogram_new = trainHistogram;
                else 
                    load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periods(testingCount)) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                    %trainHistogram_new = [trainHistogram_new; trainHistogram];
                    posDataTemp(userID).trainHistogram_new = [posDataTemp(userID).trainHistogram_new; trainHistogram];
                end
                
                
                
                oldData_expandedHistogram = [];
                for dataCount = 1:size(trainHistogram_old,1)
                    oldData_expandedHistogram = [oldData_expandedHistogram;MixHistogram(trainHistogram_old(dataCount,:))];
                end
                
                combinedData_expandedHistogram = [];
                for dataCount = 1:size(posDataTemp(userID).trainHistogram_new,1)
                    combinedData_expandedHistogram = [combinedData_expandedHistogram;MixHistogram(posDataTemp(userID).trainHistogram_new(dataCount,:))];
                end
                
                % NEGATIVE DATA
                
                numOfNegData = 40;
                
                load([sampleset_path '\' posture '\round_' num2str(round) '_train_neg_sampleSet.mat'], 'trainNegHistogram');
                
                trainNegData= [];
                rng(45);
                randNum = randperm(size(trainNegHistogram, 1));
                
                for negUserIndex = 1:numOfNegData
                    if randNum(negUserIndex) ~= userCount 
                        negData_perUser = trainNegHistogram(randNum(negUserIndex),:);
                        trainNegData = [trainNegData; negData_perUser];
                    end
                end
                
                negData_expandedHistogram = [];
                for dataCount = 1:size(trainNegData,1)
                    negData_expandedHistogram = [negData_expandedHistogram;MixHistogram(trainNegData(dataCount,:))];
                end
                
                % ~~ NEGATIVE DATA
                
                % KERNEL MAP
                
                K_value = 20;
                [clusterID, centroids, ~, distances] = kmeans(oldData_expandedHistogram, K_value);
                [oldData_centroids] = findClusterCentroid_v2(K_value, distances, clusterID, oldData_expandedHistogram);
                
                [clusterID, centroids, ~, distances] = kmeans(combinedData_expandedHistogram, K_value);
                [combinedData_centroids] = findClusterCentroid_v2(K_value, distances, clusterID, combinedData_expandedHistogram);
                
                [clusterID, centroids, ~, distances] = kmeans(negData_expandedHistogram, K_value);
                [negData_centroids] = findClusterCentroid_v2(K_value, distances, clusterID, negData_expandedHistogram);
                
                trainDataCentroids = [oldData_centroids; negData_centroids];
                
                featureWeights_old = calculateFeatureWeight_v2(K_value,oldData_expandedHistogram,negData_expandedHistogram,trainDataCentroids,size(oldData_expandedHistogram,2));
                %featureWeights_combined = calculateFeatureWeight_v2(K_value,combinedData_expandedHistogram,negData_expandedHistogram,trainDataCentroids,size(combinedData_expandedHistogram,2));
                
                featureWeights_old(featureWeights_old == 1) = 0.6;
                featureWeights_old(featureWeights_old == 0) = 0.4;
                %featureWeights_combined(featureWeight_old == 1) = 0.6;
                %featureWeights_combined(featureWeight_old == 0) = 0.4;
                
                centroidsWithWeight_old = trainDataCentroids .* featureWeights_old;
                %centroidsWithWeight_combined = trainDataCentroids_new .* featureWeights_combined;
                
                
                % CENTROID SHIFTING
                deltaMean = mean(oldData_expandedHistogram) - mean(combinedData_expandedHistogram);
                centroidsWithWeight_old_t = centroidsWithWeight_old - (repmat(deltaMean,size(centroidsWithWeight_old,1),1));
                centroidsWithWeight_old_t(centroidsWithWeight_old_t < 0) = 0;
                centroidsWithWeight_old_t(centroidsWithWeight_old_t > 1) = 1;
                % ~~~ CENTROID SHIFTING 
                
                trainData_old = [oldData_expandedHistogram; negData_expandedHistogram];
                trainData_combined = [combinedData_expandedHistogram; negData_expandedHistogram];
                
                newDataset_old = kernelMapping_v3(centroidsWithWeight_old,trainData_old,featureWeights_old);
                %newDataset_combined = kernelMapping_v3(centroidsWithWeight_old,trainData_combined,featureWeights_old);
                newDataset_combined = kernelMapping_v3(centroidsWithWeight_old_t,trainData_combined,featureWeights_old);
                 
                % ~~~ KERNEL MAP
                
                % SHIFTING
%                 deltaMean = mean(oldData_expandedHistogram) - mean(combinedData_expandedHistogram);
%                 oldData_expandedHistogram = oldData_expandedHistogram - (repmat(deltaMean,size(oldData_expandedHistogram,1),1));
%                 oldData_expandedHistogram(oldData_expandedHistogram < 0) = 0;
%                 oldData_expandedHistogram(oldData_expandedHistogram > 1) = 1;
                % ~~~ SHIFTING
                
                dataset = [oldData_expandedHistogram;combinedData_expandedHistogram];
                %dataset = [newDataset_old; newDataset_combined];
                
                % PCA
%                 [PCA_coeff_old, PCA_score_old, ~, ~, PCA_explained_old] = pca(oldData_expandedHistogram, 'NumComponents', 2);
%                 [PCA_coeff_combined, PCA_score_combined, ~, ~, PCA_explained_combined] = pca(combinedData_expandedHistogram, 'NumComponents', 2);
                [PCA_coeff, PCA_score, ~, ~, PCA_explained] = pca(dataset, 'NumComponents', 2);
                
                
                % plot/scatter
                figures = figure;
                %scatter(PCA_score_old(:,1), PCA_score_old(:,2), 300, 'filled', '^')
                scatter(PCA_score(1:40,1), PCA_score(1:40,2), 300, 'filled', '^')
                hold on
                
%                 scatter(centroidsWithWeight_old(1:end,1), centroidsWithWeight_old(1:end,2), 300, 'filled', 'c')
%                 %plot(PCA_score_old, 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 20);
%                 hold on;
%                 
                %scatter(PCA_score_combined(:,1), PCA_score_combined(:,2), 200, 'filled', 's')
                scatter(PCA_score(41:end,1), PCA_score(41:end,2), 200, 'filled', 's')
                %scatter(PCA_score(81:end-40,1), PCA_score(81:end-40,2), 200, 'filled', 's')
                %plot(PCA_score_combined, 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 20);
                legend('Old positive data', 'New positve data');
                
                set(figures, 'units', 'normalized', 'outerposition', [0 0 1 1]);
                
                %mkdir(['.\exp_result\visualization\user' num2str(userID) '\original']);
                %mkdir(['.\exp_result\visualization\user' num2str(userID) '\kernelMap']);
                %mkdir(['.\exp_result\visualization\user' num2str(userID) '\kernelMap_centroidShifted']);
                
                fileName = ['.\exp_result\visualization\user' num2str(userID) '\original\user' num2str(userID) '_testingPeriod' num2str(periods(testingCount)) '.jpg'];
                %fileName = ['.\exp_result\visualization\user' num2str(userID) '\kernelMap\user' num2str(userID) '_testingPeriod' num2str(periods(testingCount)) 'kernelMapped.jpg'];
                %fileName = ['.\exp_result\visualization\user' num2str(userID) '\kernelMap_centroidShifted\user' num2str(userID) '_testingPeriod' num2str(periods(testingCount)) 'kernelMap_centroidShifted.jpg'];
                saveas(figures,fileName);
                hold off;
                close all;
            end
        end
end