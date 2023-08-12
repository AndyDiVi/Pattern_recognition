

%imageDir = 'Crepe';
%idsCrepe = imageDatastore(imageDir,'IncludeSubfolders',true,'LabelSource', 'foldernames');
%imageLabeler(idsCrepe)
close all
clc


load('CrackRettangolo.mat');
negativeFolder = fullfile('NonCrepe');

% addestramento del detector (produzione xml di configurazione)
trainCascadeObjectDetector('my_CrackDetector.xml', gTruth ,negativeFolder,'FalseAlarmRate',0.1,'NumCascadeStages',5);

% crea il detector sulla base del file xml
detector = vision.CascadeObjectDetector('my_CrackDetector.xml');

% carica le immagini di test
idsTest = imageDatastore('CrepeTest');
numTest = numel(idsTest.Files);

for i=1:numTest
    img = readimage(idsTest,i);
    % applica il detector e trova le box
    bbox = step(detector,img);
    % aggiunge le annotazioni
    detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'Crepa');
    % visualizza
    figure(1); 
    imshow(detectedImg);
    pause
end
 
 