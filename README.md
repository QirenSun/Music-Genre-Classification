<p align="center">
<img src="https://github.com/QirenSun/Music-Generator/blob/master/Image/11-yic-pop-essay.w1200.h630.jpg" >
</p>

# Music Genre Classification   

Research in **Deep learning and machine learning**, Boston University   

## Introducation   

My work classifies ten classes music genre of a sound sample and uses Pytorch and scikit-learn to recognize the music genre.   

## GTZAN -- Datasets   

The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.    
Classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock  
Training Set: 80%  
Testing Set: 10%  
Validing Set: 10%  

## Preprocessing Data   

Mel-frequency cepstral coefficients (MFCC)  
Spectral Centroid  
Zero Crossing Rate  
Chroma Frequencies  
Spectral Roll-off  
Spectrogram images  

<p align="center">
<img src="https://github.com/QirenSun/Music-Generator/blob/master/Image/1.PNG" >
</p>

<p align="center">
<img src="https://github.com/QirenSun/Music-Generator/blob/master/Image/2.PNG" >
</p>


## Models   

Scikit-learn: SVM, MLP lbfgs(quasi-Newton method), MLP Adam(gradient-based optimizer), Decision Tree  
Pytorch: CNN, DCNN, DCNN-RNN  

<p align="center">
<img src="https://github.com/QirenSun/Music-Generator/blob/master/Image/CNN_all_layers.png" >
</p>

<p align="center">
<img src="https://github.com/QirenSun/Music-Generator/blob/master/Image/3.PNG" >
</p>


## Results  

Scikit-learn with MFCC(20), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off:  
SVM: 65.5%  
MLP adam: 66%  
MLP lbfgs: 65%  
Decision Tree: 44%  

Pytorch:  
DCNN-RNN with MFCC(50): 63%  
DCNN with MFCC(50): 60%  
CNN with spectrogram images: 43%  

## Discussion

The results of CNN, DCNN, DCNN-RNN are not the most optimized results. During analyzing the training loss and valid loss, the NN performance can increase by adjusting the learning rate, changing the parameters in the NN model, choosing the different features, and inputting the more substantial data size.  
The performance of my NN will surpass the scikit-learn results after optimization.  

