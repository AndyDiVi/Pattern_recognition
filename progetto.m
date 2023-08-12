clear all
close all
clc


imageDir = 'ImageProgetto';
ids = imageDatastore(imageDir,'IncludeSubfolders',true,'LabelSource','foldernames');
nImmagini = numel(ids.Files);
countEachLabel(ids)

% estrae una immagine su 4 per velocizzare
ids = subset(ids,1:4:nImmagini);
nImmagini = numel(ids.Files);


%legge la prima immagine per valutare la dimensione del feature vector
img = readimage(ids,8762);

[row, col, channels] = size(img); 
if (channels == 3) 
    img = rgb2gray(img);
end

%ISTOGRAMMA SCALE DI GRIGIO
figure(1)
subplot(1, 3, 1);
imshow(img, []);
fontSize = 10;
title('Original Grayscale Image', 'FontSize', fontSize);
[pixelCount , grayLevels] = imhist(img);

subplot(1, 3, 2); 
bar(pixelCount);
title('Histogram of pixels', 'FontSize', fontSize);

subplot(1,3,3);
imhist(img);
title('Histogram of original image', 'FontSize', fontSize);


%BINARIZZAZIONE
imgBW=imbinarize(img);

figure(2);
subplot(1, 3, 1); 
imshow(img);
title('Original Image', 'FontSize', fontSize);
subplot(1, 3, 2);
imshow(img, []);
title('Original Grayscale Image', 'FontSize', fontSize);
subplot(1,3,3);
imshow(imgBW);
title('Binarized Image', 'FontSize', fontSize);

% dimensione dell'operatore di HoG 
CellSize1 = [ 4 4 ];
CellSize2= [ 8 8 ];
CellSize3 = [ 12 12 ];
[hog1, visualHoG1] = extractHOGFeatures(imgBW,'CellSize',CellSize1);
[hog2, visualHoG2] = extractHOGFeatures(imgBW,'CellSize',CellSize2);
[hog3, visualHoG3] = extractHOGFeatures(imgBW,'CellSize',CellSize3);

% visualizza immagine e features
figure(3)
subplot(2,3,1:3);
imshow(imgBW);
subplot(2,3,4);
plot(visualHoG1); 
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog1))]});
subplot(2,3,5);
plot(visualHoG2); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog2))]});
subplot(2,3,6);
plot(visualHoG3); 
title({'CellSize = [12 12]'; ['Length = ' num2str(length(hog3))]});



% dimensione del vettore
fprintf('Calcolo features ... \n')
Dim_HogFeature = length(hog2);

% prepara una matrice per contenere le features del training set
Features = zeros(nImmagini, Dim_HogFeature, 'single');

for i = 1:nImmagini
    img = readimage(ids, i);
    %imgBW=imbinarize(img);
    Features(i, :) = extractHOGFeatures(img, 'CellSize', CellSize2);
end
% assegna le labels
Labels = ids.Labels;

% Estraiamo Training set e labels
XTrain = Features(1:2:end,:);
YTrain = Labels(1:2:end);
% Estraiamo Test set e labels 
XTest = Features(2:2:end,:);
YTest = Labels(2:2:end);


 
%{ 
%ANALISI PCA
%pca
Y = pdist(XTrain);
Z = linkage(Y,'ward');
T0 = cluster(Z,'maxclust',2); 
% I set 2 clasters because in my dataset is 2 classes of objects.

% PCA visualization
[W, pc] = pca(XTrain);
figure(4)
scatter(pc(:,1),pc(:,2),10,T0,'filled')
%}


%SCELTA AUTOVALORI PER PCA, computazionalmente lungo. 

A = cov(XTrain); 
L = eig(A); 
L = flip(L);
%accuratezza
Acc = cumsum(L);
Acc = Acc/sum(L);
% grafico autovalori 
figure(5)
subplot(1,2,1);
plot(L, 'r')
title('Autovalori');
subplot(1,2,2);
plot(Acc, 'r');
title('Autovalori cumulati');


%PCA
ncomp = 1000; 
coeff = pca(XTrain,'NumComponents',ncomp);
XTrain1=XTrain*coeff;
XTest1=XTest*coeff;

%{
%METODO t-SNE
fprintf('Valutazione t-SNE ... \n')

T = tsne(XTrain,'NumPCAComponents',1000);
T1 = tsne(XTrain1,'Distance','mahalanobis');
T2 = tsne(XTrain1,'Distance','chebychev');
T3 = tsne(XTrain1,'Distance','euclidean');

%SCATTER PLOT t-SNE
figure(6)
subplot(2,2,2);
gscatter(T1(:,1),T1(:,2),YTrain)
title('Mahalanobis');
subplot(2,2,3);
gscatter(T2(:,1),T2(:,2),YTrain)
title('Chebychev');
subplot(2,2,4);
gscatter(T3(:,1),T3(:,2),YTrain)
title('Euclidean');
subplot(2,2,1);
gscatter(T(:,1),T(:,2),YTrain)
title('Default');
%}

%classificatore lineare (SVM per default)
fprintf('Classificazione SVM ... \n')
classifier = fitcsvm(XTrain, YTrain);
% classifica le features di test
[predictedLinear, scoreLinear] = predict(classifier, XTest);
cp3 = classperf(cellstr(YTest), cellstr(predictedLinear));

%MATRICE DI CONFUSIONE
figure(7)
confusionchart(YTest, predictedLinear)
str = sprintf('SVM lineare - errore %.4f\n',cp3.ErrorRate);
title(str)

%{
%classificatore lineare (SVM fitclinear)
classifier2 = fitclinear(XTrain, YTrain);
[predictedLinear2, scoreLinear2] = predict(classifier2, XTest); 
cp3b = classperf(cellstr(YTest), cellstr(predictedLinear2));
fprintf('Errore test %f\n',cp3b.ErrorRate);
%MATRICE DI CONFUSIONE
figure(8)
confusionchart(YTest, predictedLinear2)
str = sprintf('SVM lineare - errore %.4f\n',cp3b.ErrorRate);
title(str)
%}

%RANDOM FORESTS
fprintf('Classificazione RF ... \n')
ntrees1 = 100;
RF_Model1 = TreeBagger(ntrees1, XTrain, YTrain, 'OOBPrediction','on'); 
view(RF_Model1.Trees{1},'Mode','graph')

%OOBERROR
figure(10)
Errore = oobError(RF_Model1);
plot(Errore);
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')
fprintf('OOBError %f\n',Errore(end));

% la applica al test
[predictedForest, scoreForest] = predict(RF_Model1, XTest); 
cp2 = classperf(cellstr(YTest), cellstr(predictedForest));
fprintf('Errore test %f\n',cp2.ErrorRate);

%matrice di confusione
figure(11)
confusionchart(YTest, categorical(predictedForest)) 
str = sprintf('Random Forest - errore %.4f\n',cp2.ErrorRate);
title(str)


% comparazione oggettiva mediante curva ROC
fprintf('Valutazione curve ROC ... \n')
[X1,Y1,T1,AUC1] = perfcurve(YTest,scoreLinear(:,2),'Positive');
AUC1 
[X2,Y2,T2,AUC2] = perfcurve(YTest,scoreForest(:,2),'Positive');
AUC2

%grafico curva
figure(12)
plot(X1,Y1,'b')
hold on
plot(X2,Y2,'r')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC curve for SVM vs RF')
legend('SVM','RF')
hold off
 
          
% mappa esempi di immagini errate (per SVM)
fprintf('Ricerca errori ... \n')
Falsi_neg = find((YTest == 'Positive')&(predictedLinear == 'Negative'));
NumFalsi_Neg = numel(Falsi_neg);
Falsi_pos = find((YTest == 'Negative')&(predictedLinear == 'Positive'));
NumFalsi_Pos = numel(Falsi_pos);

% output grafico con subplot
figure(13)
N_img = 6;
% ciclo sui falsi negativi
for k=1:min(NumFalsi_Neg,N_img)
    p = 2*Falsi_neg(k);
    img = readimage(ids, p); 
    subplot(2,N_img,k);
    imshow(img);

end
title('Falsi Negativi', 'FontSize', fontSize);



% ciclo sui falsi positivi
for k=1:min(NumFalsi_Pos,N_img)
    p = 2*Falsi_pos(k);
    img = readimage(ids, p); 
    subplot(2,N_img,k+N_img);
    imshow(img);
    
end
title('Falsi Positivi', 'FontSize', fontSize);       
          
          
          
%autoencoder

%{
%training MLP
fprintf('Classificazione con MLP ...\n');

%prepara i dati

net_labels = full(cellstr(YTrain));
% numero di nodi hidden (sperimentare)
nhidden = 200;

% crea ed addestra la rete
net = patternnet(nhidden);
%net = train(net,cellstr(XTrain)', net_labels);
autoenc = trainAutoencoder(XTrain,nhidden);

view(autoenc)

figure(20)
plotWeights(autoenc);

feat1 = encode(autoenc,XTrain);

hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
view(autoenc2)
feat2 = encode(autoenc2,feat1);
figure(20)
plotWeights(autoenc2);

softnet = trainSoftmaxLayer(feat2, XTrain,'MaxEpochs',400);
view(softnet)

stackednet = stack(autoenc,autoenc2,softnet);
view(stackednet)

imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Load the test images
[XTrain,tTest] = digitTestCellArrayData;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end


y = stackednet(xTest);
plotconfusion(tTest,y);
y = stackednet(cellstr(YTest));
figure(21)
plotconfusion(XTrain,y);
confusionchart(YTest, y);


% applica al test
test_out = net(XTest');
test_out2 = autoenc(XTest);
test_net = vec2ind(test_out)'-1;
test_net2 = vec2ind(test_out2)'-1;

% oggetto per valutazione prestazioni
cp5 = classperf(cellstr(YTest), cellstr(test_net));
cp5_2 = classperf(cellstr(YTest), cellstr(test_net2));
fprintf('Errore test %f\n',cp5.ErrorRate);
 statistica errori
figure(22)
confusionchart(YTest, test_net);
str = sprintf('MultiLayer Net - errore %.4f\n',cp5.ErrorRate);
title(str)
%}
