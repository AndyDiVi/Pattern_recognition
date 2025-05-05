# 🧠 Crack Detection – Pattern Recognition Project

Progetto accademico per il **minicorso di Pattern Recognition** (corso di *Metodi Predittivi per l’Azienda*, a.a. 2022/2023), incentrato sul rilevamento automatico di **crepe strutturali** attraverso tecniche di elaborazione delle immagini, estrazione di feature e classificazione.

Le crepe superficiali sono indicatori critici di degrado strutturale e la loro corretta identificazione è essenziale per garantire sicurezza e manutenzione tempestiva. L’obiettivo di questo progetto è la realizzazione di un **sistema automatico di crack detection**, come alternativa più efficiente all’ispezione visiva manuale.

## 🛠 Tecniche Utilizzate

- **Segmentazione e Binarizzazione** delle immagini per l’evidenziazione delle crepe  
- **Zonatura** e **estrazione di feature HOG**  
- **Riduzione dimensionale** (PCA e t-SNE)  
- **Classificazione** con algoritmi SVM e Random Forest  
- **Detection automatica** tramite Image Labeler e `trainCascade` di MATLAB
  
# Getting Started
## 📦 Contenuto del Repository

- `Detection.m` – Script per la rilevazione automatica delle crepe  
- `progetto.m` – Script principale per preprocessing, segmentazione ed estrazione delle feature  
- `CrackPixel.mat`, `CrackRettangolo.mat` – Definizione delle **regioni di interesse (ROI)** create manualmente per la fase di detection (dataset immagini di crepe da Kaggle)  
- `my_CrackDetector.xml` – Classificatore pre-addestrato (output di `trainCascade`)  
- `CrackProject.pdf` – Relazione tecnica con descrizione del progetto, delle fasi e delle metodologie  
- `README.md` – Questo file

## ▶️ Esecuzione

1. Assicurati di avere installato **MATLAB** con l’Image Processing Toolbox.  
2. Avvia `progetto.m` per eseguire la pipeline completa di classificazione.  
3. Utilizza `Detection.m` per eseguire la rilevazione automatica tramite il classificatore addestrato.  
4. Il file `.xml` può essere usato direttamente con `vision.CascadeObjectDetector`.
