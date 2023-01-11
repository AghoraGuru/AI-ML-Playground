% making cnn with Deep Learning Tool Box
%loading a pickle label file
load('RML2016.10a_dict.pkl');
% finding modulation types
modulation_types = fieldnames(dict);

%making a matrix for modulation types
modulation_types_matrix = zeros(1,11);
for i = 1:11
    modulation_types_matrix(i) = i;
end
%splitting data into train and test

%defining xtrain and ytrain
XTrain = [];
YTrain = [];
% adding data to xtrain and ytrain
for i = 1:11
    for j = 1:100
        XTrain = [XTrain; dict.(modulation_types{i})(j,:)];
        YTrain = [YTrain; modulation_types_matrix(i)];
    end
end
% reshaping xtrain
XTrain = reshape(XTrain,[2 128 1 1100]);
% making ytrain categorical
YTrain = categorical(YTrain);

%defining xtest and ytest
XTest = [];
YTest = [];
% adding data to xtest and ytest
for i = 1:11
    for j = 101:120
        XTest = [XTest; dict.(modulation_types{i})(j,:)];
        YTest = [YTest; modulation_types_matrix(i)];
    end
end
% reshaping xtest
XTest = reshape(XTest,[2 128 1 220]);
% making ytest categorical
YTest = categorical(YTest);


%making a cnn
layers = [imageInputLayer([2 128 1])
          convolution2dLayer(3,8,'Padding','same')
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer(3,16,'Padding','same')
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer(3,32,'Padding','same')
          batchNormalizationLayer
          reluLayer
          fullyConnectedLayer(11)
          softmaxLayer
          classificationLayer];

%training the cnn
options = trainingOptions('sgdm','MaxEpochs',10,'InitialLearnRate',0.0001);
net = trainNetwork(XTrain,YTrain,layers,options);

%testing the cnn
YPred = classify(net,XTest);
accuracy = sum(YPred == YTest)/numel(YTest)

%plotting confusion matrix
plotconfusion(YTest,YPred)
