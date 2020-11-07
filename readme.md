# Change Detection Based on Artificial Intelligence: State-of-the-Art and Challenges

## 1. Introduction
 Change detection based on remote sensing (RS) data is an important method of detecting changes on the Earth’s surface and has a wide range of applications in urban planning, environmental monitoring, agriculture investigation, disaster assessment, and map revision. In recent years, integrated artificial intelligence (AI) technology has become a research focus in developing new change detection methods. Although some researchers claim that AI-based change detection approaches outperform traditional change detection approaches, it is not immediately obvious how and to what extent AI can improve the performance of change detection. This review focuses on the state-of-the-art methods, applications, and challenges of AI for change detection. Specifically, the implementation process of AI-based change detection is first introduced. Then, the data from different sensors used for change detection, including optical RS data, synthetic aperture radar (SAR) data, street view images, and combined heterogeneous data, are presented, and the available open datasets are also listed. The general frameworks of AI-based change detection methods are reviewed and analyzed systematically, and the unsupervised schemes used in AI-based change detection are further analyzed. Subsequently, the commonly used networks in AI for change detection are described. From a practical point of view, the application domains of AI-based change detection methods are classified based on their applicability. Finally, the major challenges and prospects of AI for change detection are discussed and delineated, including (a) heterogeneous big data processing, (b) unsupervised AI, and (c) the reliability of AI. This review will be beneficial for researchers in understanding this field.

![](/Figure%201.png)
<center>Figure 1. General schematic diagram of change detection.</center>

## 2. Implementation process

Figure 2 provide a general implementation process of AI-based change detection, but the structure of the AI model is diverse and needs to be well designed according to different application situations and the training data. It is worth mentioning that existing mature frameworks such as <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a>, <a href="https://keras.io/" target="_blank">Keras</a>, <a href="https://pytorch.org/" target="_blank">Pytorch</a>, and <a href="https://caffe.berkeleyvision.org/" target="_blank">Caffe</a>, help researchers more easily realize the design, training, and deployment of AI models, and their development documents provide detailed introductions.

![](/Figure%202.png)
<center>Figure 2. Implementation process of AI-based change detection (black arrows indicate workflow and red arrow indicates an example).</center>

### 2.1 Available codes for AI-based methods

<table>
<caption>Table 1. A list of available codes for AI-based change detection methods.</caption>
	<tr>
	    <th>Methods</th>
	    <th>Keywords</th>
	    <th>Publication</th>  
        <th>(Re-)Implementation</th>
	</tr>
	<tr>
	    <td>DSMSCN</td>
	    <td>CNN; Siamese; Multi-scale; Unsupervised/Supervised; Optical RS</td>
	    <td> A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sening images, arXiv, 2020.  [<a href="https://arxiv.org/abs/1906.11479" target="_blank">paper</a>], [<a href="https://github.com/I-Hope-Peace/DSMSCN" target="_blank">code, dataset</a>]</td>
        <td>Tensorflow 1.9</td>
	</tr>
	<tr>
	    <td>SiamCRNN</td>
	    <td>CNN+RNN; Siamese; Multi-source; Optical RS</td>
	    <td> Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network, TGRS, 2020.  [<a href="https://doi.org/10.1109/TGRS.2019.2956756" target="_blank">paper</a>], [<a href="https://github.com/I-Hope-Peace/SiamCRNN" target="_blank">code, dataset</a>]</td>
        <td>Tensorflow 1.9</td>
	</tr>
	<tr>
	    <td>DSIFN</td>
	    <td>CNN; Attention Mechanism; Optical RS</td>
	    <td> A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sening images, ISPRS, 2020.  [<a href="https://doi.org/10.1016/j.isprsjprs.2020.06.003" target="_blank">paper</a>], [<a href="https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images" target="_blank">code, dataset</a>]</td>
        <td>Pytorch & Keras</td>
	</tr>
	<tr>
	    <td>CEECNet</td>
	    <td>CNN; Attention Mechanism; Similarity Measure; Optical RS</td>
	    <td> Looking for change? Roll the Dice and demand Attention, arXiv, 2020.  [<a href="https://arxiv.org/abs/2009.02062" target="_blank">paper</a>], [<a href="https://github.com/feevos/ceecnet" target="_blank">code, dataset</a>]</td>
        <td>MXNet + Python</td>
	</tr>
	<tr>
	    <td>LamboiseNet</td>
	    <td>CNN (Light UNet++); Optical RS</td>
	    <td> Change detection in satellite imagery using deep learning, Master Thesis.  [<a href="https://github.com/hbaudhuin/LamboiseNet" target="_blank">code, dataset, pre-trained model</a>]</td>
        <td>Pytorch</td>
	</tr>
	<tr>
	    <td>DTCDSCN</td>
	    <td>CNN; Siamese</td>
	    <td> Building change detection for remote sensing images using a dual task constrained deep siamese convolutional network model, undergoing review.  [<a href="https://github.com/fitzpchao/DTCDSCN" target="_blank">code, dataset</a>]</td>
        <td>Pytorch</td>
	</tr>
	<tr>
	    <td>Land-Cover-Analysis</td>
	    <td>CNN (UNet); Post-Classification;  Optical RS</td>
	    <td> Land Use/Land cover change detection in cyclone affected areas using convolutional neural networks.  [<a href="https://github.com/Kalit31/Land-Cover-Analysis/blob/master/Report.pdf" target="_blank">report</a>], [<a href="https://github.com/Kalit31/Land-Cover-Analysis" target="_blank">code, dataset, pre-trained model</a>]</td>
        <td>TensorFlow+Keras</td>
	</tr>
	<tr>
	    <td>CorrFusionNet</td>
	    <td>CNN; Scene-level; Siamese;  Optical RS</td>
	    <td> Correlation based fusion network towards multi-temporal scene classification and change detection, undergoing review.  [<a href="https://github.com/rulixiang/CorrFusionNet" target="_blank">code, pre-trained model</a>], [<a href="https://github.com/rulixiang/MtS-WH-Dataset" target="_blank">dataset</a>]</td>
        <td>TensorFlow 1.8</td>
	</tr>
	<tr>
	    <td>SSCDNet</td>
	    <td>CNN (ResNet18); Siamese; Transfer Learning; Semantic; Streetview</td>
	    <td>Weakly supervised silhouette-based semantic scene change detection, ICRA, 2020.  [<a href="https://arxiv.org/abs/1811.11985" target="_blank">paper</a>] [<a href="https://github.com/xdspacelab/sscdnet" target="_blank">code, dataset, pre-trained model</a>]</td>
        <td>Pytorch+Python3.6</td>
	</tr>
     <tr>
	    <td>Heterogeneous_CD</td>
	    <td>AE (Code-Aligned AE); Unsupervised; Transformation; Heterogeneous; Optical RS</td>
	    <td>Code-aligned autoencoders for unsupervised change detection in multimodal remote sensing images, arXiv, 2020. [<a href="https://arxiv.org/abs/2004.07011" target="_blank">paper</a>]  [<a href="https://github.com/llu025/Heterogeneous_CD/tree/master/Code-Aligned_Autoencoders" target="_blank">code, dataset</a>]</td>
        <td>TensorFlow 2.0</td>
	</tr>
    <tr>
	    <td>FDCNN</td>
	    <td>CNN (VGG16); Transfer Learning; Pure-Siamese; Multi-scale; Optical RS</td>
	    <td>A feature difference convolutional neural network-based change detection method, TGRS, 2020. [<a href="https://dx.doi.org/10.1109/tgrs.2020.2981051" target="_blank">paper</a>]  [<a href="https://github.com/MinZHANG-WHU/FDCNN" target="_blank">code, dataset, pre-trained model</a>]</td>
        <td>Caffe+Python2.7</td>
	</tr>
    <tr>
	    <td>STANet</td>
	    <td>CNN (ResNet-18); Attention Mechanism; Pure-Siamese; Spatial–Temporal Dependency; Optical RS</td>
	    <td>A spatial-temporal attention-based method and a new dataset for remote sensing image change detection, RS, 2020. [<a href="https://dx.doi.org/10.3390/rs12101662" target="_blank">paper</a>]  [<a href="https://github.com/justchenhao/STANet" target="_blank">code, dataset</a>]</td>
        <td>Pytorch+Python3.6</td>
	</tr>
    <tr>
	    <tr>
	    <td>X-Net</td>
	    <td>CNN; Unsupervised; Transformation; Heterogeneous; Optical RS; SAR</td>
	    <td>Deep image translation with an affinity-based change prior for unsupervised multimodal change detection, 2020. [<a href="https://arxiv.org/abs/2001.04271" target="_blank">paper</a>]  [<a href="https://github.com/llu025/Heterogeneous_CD/tree/master/legacy/Deep_Image_Translation" target="_blank">code, dataset</a>]</td>
        <td>Tensorflow 1.4</td>
	</tr>
    <tr>
	    <tr>
	    <td>ACE-Net</td>
	    <td>AE (Adversarial Cyclic Encoders); Unsupervised; Transformation; Heterogeneous; Optical RS; SAR</td>
	    <td>Deep image translation with an affinity-based change prior for unsupervised multimodal change detection, 2020. [<a href="https://arxiv.org/abs/2001.04271" target="_blank">paper</a>]  [<a href="https://github.com/llu025/Heterogeneous_CD/tree/master/legacy/Deep_Image_Translation" target="_blank">code, dataset</a>]</td>
        <td>Tensorflow 1.4</td>
	</tr>
     <tr>
	    <td>VGG_LR</td>
	    <td>CNN (VGG16); Transfer Learning; Pure-Siamese; SLIC; Low Ranks; Optical RS</td>
	    <td>Change detection based on deep features and low rank, GRSL, 2017. [<a href="https://dx.doi.org/10.1109/LGRS.2017.2766840" target="_blank">paper</a>]  [<a href="https://github.com/MinZHANG-WHU/FDCNN/tree/master/vgg_lr" target="_blank">re-implementation code, dataset, pre-trained model</a>]</td>
        <td>Caffe+Matlab</td>
	</tr>
	<tr>
	    <td>CDNet</td>
	    <td>CNN; Siamese; Multimodal Data; Point Cloud Data</td>
	    <td> Detecting building changes between airborne laser scanning and photogrammetric data, RS, 2019. [<a href="https://doi.org/10.3390/rs11202417" target="_blank">paper</a>], [<a href="https://github.com/Zhenchaolibrary/PointCloud2PointCloud-Change-Detection" target="_blank">code</a>]</td>
        <td>Pytorch</td>
	</tr>
     <tr>
	    <td>SCCN</td>
	    <td>AE (DAE); Unsupervised; Heterogeneous; Optical RS; SAR</td>
	    <td>A deep convolutional coupling network for change detection based on heterogeneous optical and radar images, TNNLS, 2018. [<a href="https://dx.doi.org/10.1109/TNNLS.2016.2636227" target="_blank">paper</a>]  [<a href="https://github.com/llu025/Heterogeneous_CD/tree/master/Code-Aligned_Autoencoders" target="_blank">re-implementation code</a>]</td>
        <td>TensorFlow 2.0</td>
	</tr>
    <tr>
	    <td>cGAN</td>
	    <td>GAN (conditional GAN); Heterogeneous; Optical RS; SAR</td>
	    <td> A conditional adversarial network for change detection in heterogeneous images, GRSL, 2019. [<a href="https://dx.doi.org/10.1109/LGRS.2018.2868704" target="_blank">paper</a>]  [<a href="https://github.com/llu025/Heterogeneous_CD/tree/master/Code-Aligned_Autoencoders" target="_blank">re-implementation code</a>]</td>
        <td>TensorFlow 2.0</td>
	</tr>
    <tr>
	    <td>DASNet</td>
	    <td>CNN (VGG16); Siamese; Attention Mechanism  ; Optical RS</td>
	    <td>DASNet: Dual attentive fully convolutional siamese networks for change detection of high resolution satellite images, arXiv, 2020. [<a href="" target="_blank">paper</a>]  [<a href="https://github.com/lehaifeng/DASNet" target="_blank">code, dataset, pre-trained model</a>]</td>
        <td>Pytorch+Python3.6</td>
	</tr>
    <tr>
	    <td>UNetLSTM</td>
	    <td>CNN (UNet); RNN (LSTM); Integrated Model; Optical RS</td>
	    <td>Detecting Urban Changes With Recurrent Neural Networks From Multitemporal Sentinel-2 Data, IGARSS, 2019. [<a href="https://arxiv.org/abs/1910.07778" target="_blank">paper</a>]  [<a href="https://github.com/granularai/chip_segmentation_fabric" target="_blank">code, dataset, pre-trained model</a>] and  [<a href="https://github.com/SebastianHafner/urban_change_detection" target="_blank">code</a>]</td>
        <td>Pytorch+Python3.6</td>
	</tr><tr>
	    <td>CDMI-Net</td>
	    <td>CNN (Unet); Pure-Siamese; Multiple Instance Learning; Landslide Mapping; Optical RS</td>
	    <td>Deep multiple instance learning for landslide mapping, GRSL, 2020. [<a href="https://dx.doi.org/10.1109/LGRS.2020.3007183" target="_blank">paper</a>]  [<a href="https://github.com/MinZHANG-WHU/CDMI-Net" target="_blank">code, pre-trained model</a>]</td>
        <td>Pytorch+Python3.6</td>
	</tr>
    <tr>
	    <td>DSFANet</td>
	    <td>DNN; Unsupervised; Pre-classification; Slow Feature Analysis; Optical RS</td>
	    <td>Unsupervised deep slow feature analysis for change detection in multi-temporal remote sensing images, TGRS, 2019. [<a href="https://dx.doi.org/10.1109/TGRS.2019.2930682" target="_blank">paper</a>]  [<a href="https://github.com/rulixiang/DSFANet" target="_blank">code, dataset</a>]</td>
        <td>TensorFlow 1.7</td>
	</tr>
    <tr>
	    <td>CD-UNet++</td>
	    <td>CNN (improved UNet++); Direct Classification; Optical RS</td>
	    <td>End-to-end change detection for high resolution satellite images using improved UNet++, RS, 2019. [<a href="https://doi.org/10.3390/rs11111382" target="_blank">paper</a>]  [<a href="https://github.com/daifeng2016/End-to-end-CD-for-VHR-satellite-image" target="_blank">code</a>]</td>
        <td>TensorFlow+Keras</td>
	</tr>
    <tr>
	    <td>SiameseNet</td>
	    <td>CNN (VGG16); Pure-Siamese; Optical RS</td>
	    <td>Siamese network with multi-level features for patch-based change detection in satellite imagery, GlobalSIP, 2018. [<a href="https://sigport.org/documents/siamese-network-multi-level-features-patch-based-change-detection-satellite-imagery" target="_blank">paper</a>]  [<a href="https://github.com/vbhavank/Siamese-neural-network-for-change-detection" target="_blank">code, dataset</a>]</td>
        <td>TensorFlow+Keras</td>
	</tr>
	<tr>
	    <td>Re3FCN</td>
	    <td>CNN (ConvLSTM); PCA; 3D convolution; Multi-class changes; Optical RS; Hyperspectral</td>
	    <td>Change detection in hyperspectral images using recurrent 3D fully convolutional networks, RS, 2018. [<a href="https://doi.org/10.3390/rs10111827" target="_blank">paper</a>]  [<a href="https://github.com/mkbensalah/Change-Detection-in-Hyperspectral-Images target="_blank">code, dataset</a>]</td>
        <td>TensorFlow+Keras</td>
	</tr>
    <tr>
	    <td>FC-EF, FC-Siam-conc, FC-Siam-diff</td>
	    <td>CNN (UNet); Pure-Siamese; Optical RS</td>
	    <td>Fully convolutional siamese networks for change detection, ICIP, 2018. [<a href="https://arxiv.org/abs/1810.08462" target="_blank">paper</a>]  [<a href="https://github.com/rcdaudt/fully_convolutional_change_detection" target="_blank">code, dataset</a>]</td>
        <td>Pytorch</td>
	</tr>
    <tr>
	    <td>CosimNet</td>
	    <td>CNN (Deeplab v2); Pure-Siamese; Streetview</td>
	    <td>Learning to measure changes: fully convolutional siamese metric networks for scene change detection, arXiv, 2018. [<a href="https://arxiv.org/abs/1810.09111" target="_blank">paper</a>]  [<a href="https://github.com/gmayday1997/SceneChangeDet" target="_blank">code, dataset, pre-trained model</a>]</td>
        <td>Pytorch+Python2.7</td>
	</tr>
    <tr>
	    <td>Mask R-CNN</td>
	    <td>Mask R-CNN (ResNet-101); Transfer Learning; Post-Classification; Optical RS </td>
	    <td>Slum segmentation and change detection: a deep learning approach, NIPS, 2018. [<a href="https://arxiv.org/abs/1811.07896" target="_blank">paper</a>]  [<a href="https://github.com/cbsudux/Mumbai-slum-segmentation" target="_blank">code, dataset, pre-trained model</a>]</td>
        <td>TensorFlow+Keras</td>
	</tr>
    <tr>
	    <td>CaffeNet</td>
	    <td>CNN (CaffeNet); Unsupervised; Transfer Learning; Optical RS</td>
	    <td>Convolutional neural network features based change detection in satellite images, IWPR, 2016. [<a href="https://doi.org/10.1117/12.2243798" target="_blank">paper</a>]  [<a href="https://github.com/vbhavank/Unstructured-change-detection-using-CNN" target="_blank">code, dataset</a>]</td>
        <td>TensorFlow+Keras</td>
	</tr>
    <tr>
	    <td>CWNN</td>
	    <td>CNN (CWNN); Unsupervised; Pre-Classification; SAR</td>
	    <td>Sea ice change detection in SAR images based on convolutional-wavelet neural networks, GRSL, 2019. [<a href="https://dx.doi.org/10.1109/LGRS.2019.2895656" target="_blank">paper</a>]  [<a href="https://github.com/summitgao/SAR_Change_Detection_CWNN" target="_blank">code, dataset</a>]</td>
        <td>Matlab</td>
	</tr>
    <tr>
	    <td>MLFN</td>
	    <td>CNN (DenseNet); Transfer learning; SAR</td>
	    <td>Transferred deep learning for sea ice change detection from synthetic aperture radar images, GRSL, 2019. [<a href="https://dx.doi.org/10.1109/LGRS.2019.2906279" target="_blank">paper</a>]  [<a href="https://github.com/summitgao/SAR-Change-Detection-MLFN" target="_blank">code, dataset</a>]</td>
        <td>Caffe+Matlab</td>
	</tr>
    <tr>
	    <td>GarborPCANet</td>
	    <td>CNN (PCANet); Unsupervised; Pre-Classification; Gabor Wavelets; SAR</td>
	    <td>Automatic change detection in synthetic aperture radar images based on PCANet, GRSL, 2016. [<a href="https://dx.doi.org/10.1109/LGRS.2016.2611001" target="_blank">paper</a>]  [<a href="https://github.com/summitgao/SAR_Change_Detection_GarborPCANet" target="_blank">code, dataset</a>]</td>
        <td>Matlab</td>
	</tr>
    <tr>
	    <td>Ms-CapsNet</td>
	    <td>CNN (Ms-CapsNet); Capsule; Attention Mechanism; Adaptive Fusion Convolution; SAR</td>
	    <td>Change detection in SAR images based on multiscale capsule network, GRSL, 2020. [<a href="https://dx.doi.org/10.1109/LGRS.2020.2977838" target="_blank">paper</a>]  [<a href="https://github.com/summitgao/SAR_CD_MS_CapsNet" target="_blank">code, dataset</a>]</td>
        <td>Matlab+Keras2.16</td>
	</tr>
    <tr>
	    <td>DCNet</td>
	    <td>CNN; Unsupervised; Pre-Classification; SAR</td>
	    <td>Change detection from synthetic aperture radar images based on channel weighting-based deep cascade network, JSTARS, 2019. [<a href="https://dx.doi.org/10.1109/JSTARS.2019.2953128" target="_blank">paper</a>]  [<a href="https://github.com/summitgao/SAR_CD_DCNet" target="_blank">code, dataset</a>]</td>
        <td>Caffe</td>
	</tr>
	<tr>
	    <td>ChangeNet</td>
	    <td>CNN; Siamese; StreetView</td>
	    <td>ChangeNet: a deep learning architecture for visual change detection, ECCV, 2018. [<a href="http://openaccess.thecvf.com/content_eccv_2018_workshops/w7/html/Varghese_ChangeNet_A_Deep_Learning_Architecture_for_Visual_Change_Detection_ECCVW_2018_paper.html" target="_blank">paper</a>]  [<a href="https://github.com/leonardoaraujosantos/ChangeNet" target="_blank">code, dataset</a>]</td>
        <td>Pytorch</td>
	</tr>
    <tr>
	    <td colspan="4">Others will be added soon!</td>
    </tr>
</table>

### 2.2 Available codes for traditional methods

<table>
<caption>Table 2. A list of available codes for traditional change detection methods.</caption>
	<tr>
	    <th>Methods</th>
	    <th>Keywords</th>
	    <th>Publication</th>  
        <th>Implementation</th>  
	</tr>
     <tr>
	    <td>Several Classical Methods</td>
	    <td>CVA; DPCA; Image Differencing; Image Ratioing; Image Regression; IR-MAD; MAD; PCAkMeans; PCDA; KMeans; OTSU; Fixed Threshold</td>
	    <td>A toolbox for remote sensing change detection. [<a href="https://github.com/Bobholamovic/ChangeDetectionToolbox" target="_blank">code</a>]</td>
        <td>Matlab</td>
	</tr>
	<tr>
	    <td>Matlab Toolbox Change Detection</td>
	    <td>IR-MAD; IT-PCA; ERM; ICM</td>
	    <td>A toolbox for unsupervised change detection analysis, IJRS, 2016.[<a href="https://doi.org/10.1080/01431161.2016.1154226" target="_blank">paper</a>] [<a href="https://github.com/NicolaFalco/Matlab-toolbox-change-detection" target="_blank">code</a>]</td>
        <td>Matlab</td>
	</tr>
    <tr>
	    <td>RFR,SVR,GPR</td>
	    <td>Unsupervised; Image Regression; Heterogeneous; Optical RS; SAR</td>
	    <td>Unsupervised image regression for heterogeneous change detection, TGRS, 2019. [<a href="https://dx.doi.org/10.1109/TGRS.2019.2930348" target="_blank">paper</a>]  [<a href="https://github.com/llu025/Heterogeneous_CD/tree/master/legacy/Image_Regression" target="_blank">code</a>]</td>
        <td>Matlab</td>
	</tr>
     <tr>
	    <td>HPT</td>
	    <td>Unsupervised; Transformation; Heterogeneous; Optical RS; SAR</td>
	    <td>Change detection in heterogenous remote sensing images via homogeneous pixel transformation, TIP, 2018. [<a href="https://dx.doi.org/10.1109/TIP.2017.2784560" target="_blank">paper</a>]  [<a href="https://github.com/llu025/Heterogeneous_CD/tree/master/legacy/Image_Regression" target="_blank">re-implementation code</a>]</td>
        <td>Matlab</td>
	</tr>
     <tr>
	    <td>kCCA</td>
	    <td>Canonical Correlation Analysis; Cross-Sensor; Optical RS</td>
	    <td>Spectral alignment of multi-temporal cross-sensor images with automated kernel correlation analysis, IJPRS, 2015. [<a href="https://doi.org/10.1016/j.isprsjprs.2015.02.005" target="_blank">paper</a>]  [<a href="https://sites.google.com/site/michelevolpiresearch/codes/cross-sensor" target="_blank">code</a>]</td>
        <td>Matlab</td>
	</tr>
    <tr>
	    <td>Ker. Diff. RBF</td>
	    <td>Unsupervised; K-means; Optical RS</td>
	    <td>Unsupervised change detection with kernels, GRSL, 2012. [<a href="https://dx.doi.org/10.1016/j.jag.2011.10.013" target="_blank">paper</a>]  [<a href="https://drive.google.com/file/d/0B9xP9Y5JKJz0Q1ctbDJERWpTd2s/edit?usp=sharing" target="_blank">code</a>]</td>
        <td>Matlab</td>
	</tr>
    <tr>
	    <td>FDA-RM</td>
	    <td>DI-based; Frequency-Domain Analysis; Random Multigraphs; SAR</td>
	    <td>Synthetic aperture radar image change detection based on frequency domain analysis and random multigraphs, JARS, 2018. [<a href="https://doi.org/10.1117/1.JRS.12.016010" target="_blank">paper</a>]  [<a href="https://github.com/summitgao/SAR_Change_Detection_FDA_RMG" target="_blank">code</a>]</td>
        <td>Matlab </td>
	</tr>
    <tr>
	    <td>CD-NR-ELM</td>
	    <td>DI-based; Pre-Classification; Extreme Learning Machine; SAR</td>
	    <td>Change detection from synthetic aperture radar images based on neighborhood-based ratio and extreme learning machine, JARS, 2016. [<a href="https://doi.org/10.1117/1.JRS.10.046019" target="_blank">paper</a>]  [<a href="https://github.com/summitgao/SAR_Change_Detection_NR_ELM" target="_blank">code, dataset</a>]</td>
        <td>Matlab</td>
	</tr>
     <tr>
	    <td>None</td>
	    <td>Likelihood Ratio; Test Statistic; SAR</td>
	    <td>Change detection in polarimetric SAR
images, 2015. [<a href="https://github.com/fouronnes/SAR-change-detection/blob/master/SAR_Change_Detection_Victor_Poughon.pdf" target="_blank">report</a>]  [<a href="https://github.com/fouronnes/SAR-change-detection" target="_blank">code</a>]</td>
        <td>Python</td>
	</tr>
	<tr>
	    <td>PCA K-Means</td>
	    <td>Unsupervised; DI-based; PCA; K Means; Optical RS</td>
	    <td>Unsupervised Change Detection in Satellite Images Using Principal Component Analysis and k-Means Clustering, GRSL, 2009. [<a href="https://dx.doi.org/10.1109/LGRS.2009.2025059" target="_blank">paper</a>]  [<a href="https://github.com/rulixiang/ChangeDetectionPCAKmeans" target="_blank">re-implementation code, dataset</a>] or [<a href="https://github.com/leduckhai/Change-Detection-PCA-KMeans" target="_blank">re-implementation code</a>]</td>
        <td>Matlab</td>
	</tr>
    <tr>
	    <td colspan="4">Others will be added soon!</td>
    </tr>
</table>


## 3. Open datasets

Currently, there are some freely available data sets for change detection, which can be used as benchmark datasets for AI training and accuracy evaluation in future research. Detailed information is presented in Table 3.
<table>
<caption>Table 3. A list of open datasets for change detection.</caption>
	<tr>
	    <th>Type</th>
	    <th width="180px">Data set</th>
	    <th>Description</th>  
	</tr>
	<tr>
	    <td rowspan="16">Optical RS</td>
	    <td>SEmantic Change detectiON Dataset (SECOND) [<a href="#Ref-24">24</a>] </td>
	    <td>a pixel-level annotated semantic change detection dataset, including 4662 pairs of aerial images with 512 x 512 pixels from several platforms and sensors, covering Hangzhou, Chengdu, and Shanghai.  It focus on 6 main land-cover classes, i.e. , non-vegetated ground surface, tree, low vegetation, water, buildings and playgrounds , that are frequently involved in natural and man-made geographical changes. [<a href="http://www.captain-whu.com/PROJECT/SCD/" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>Hyperspectral change detection dataset [<a href="#Ref-1">1</a>] </td>
	    <td>3 different hyperspectral scenes acquired by AVIRIS or HYPERION sensor, with 224 or 242 spectral bands, labeled 5 types of changes related with crop transitions at pixel level. [<a href="https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>River HSIs dataset  [<a href="#Ref-2">2</a>] </td>
	    <td>2 HSIs in Jiangsu province, China, with 198 bands, labeled as changed and unchanged at pixel level. [<a href="https://drive.google.com/file/d/1cWy6KqE0rymSk5-ytqr7wM1yLMKLukfP/view" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>HRSCD  [<a href="#Ref-3">3</a>]</td>
	    <td>291 co-registered pairs of RGB aerial images, with pixel-level change and land cover annotations, providing hierarchical level change labels, for example, level 1 labels include five classes: no information, artificial surfaces, agricultural areas, forests, wetlands, and water. [<a href="https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>WHU building dataset  [<a href="#Ref-4">4</a>]</td>
	    <td>2-period aerial images containing 12,796 buildings, provided along with building vector and raster maps. [<a href="http://study.rsgis.whu.edu.cn/pages/download/" target="_blank">Download</a>]</td>
	</tr>
	<tr><td>SZTAKI Air change benchmark  [<a href="#Ref-5">5</a>, <a href="#Ref-6">6</a>]</td>
	    <td>13 aerial image pairs with 1.5 m spatial resolution, labeled as changed and unchanged at pixel level. [<a href="http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>OSCD  [<a href="#Ref-7">7</a>]</td>
	    <td>24 pairs of multispectral images acquired by Sentinel-2, labeled as changed and unchanged at pixel level. [<a href="https://rcdaudt.github.io/oscd/" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>Change detection dataset  [<a href="#Ref-8">8</a>]</td>
	    <td>4 pairs of multispectral images with different spatial resolutions, labeled as changed and unchanged at pixel level. [<a href="https://github.com/MinZHANG-WHU/FDCNN" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>MtS-WH [<a href="#Ref-9">9</a>]</td>
	    <td>2 large-size VHR images acquired by IKONOS sensors, with 4 bands and 1 m spatial resolution, labeled 5 types of changes (i.e., parking, sparse houses, residential region, and vegetation region) at scene level. [<a href="http://sigma.whu.edu.cn/newspage.php?q=2019_03_26" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>ABCD  [<a href="#Ref-10">10</a>]</td>
	    <td>16,950 pairs of RGB aerial images for detecting washed buildings by tsunami, labeled damaged buildings at scene level. [<a href="https://github.com/gistairc/ABCDdataset" target="_blank">Download</a>]</td>
	</tr>
    <tr>
	    <td>xBD  [<a href="#Ref-11">11</a>]</td>
	    <td>Pre- and post-disaster satellite imageries for building damage assessment, with over 850,000 building polygons from 6 disaster types, labeled at pixel level with 4 damage scales. [<a href="https://xview2.org/dataset" target="_blank">Download</a>]</td>
    </tr>
    <tr>
	    <td>AICD  [<a href="#Ref-12">12</a>]</td>
	    <td>1000 pairs of synthetic aerial images with artificial changes generated with a rendering engine, labeled as changed and unchanged at pixel level. [<a href="https://computervisiononline.com/dataset/1105138664" target="_blank">Download</a>]</td>
    </tr>
    <tr>
	    <td>Database of synthetic and real images  [<a href="#Ref-13">13</a>]</td>
	    <td>24,000 synthetic images and 16,000 fragments of real season-varying RS images obtained by Google Earth, labeled as changed and unchanged at pixel level. [<a href="https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit" target="_blank">Download</a>]</td>
    </tr>
    <tr>
	    <td>LEVIR-CD  [<a href="#Ref-14">14</a>]</td>
	    <td>637 very high-resolution (VHR, 0.5m/pixel) Google Earth (GE) image patch pairs with a size of 1024 × 1024 pixels and contains a total of 31,333 individual change building instances, labeled as changed and unchanged at pixel level. [<a href="https://justchenhao.github.io/LEVIR/" target="_blank">Download</a>]</td>
    </tr>
    <tr>
	    <td>Bastrop fire dataset [<a href="#Ref-21">21</a>]</td>
	    <td>4 images acquired by different sensors over the Bastrop County, Texas (USA). It is composed by a Landsat 5 TM as the pre-event image and a Landsat 5 TM, a EO-1 ALI and a Landsat 8 as post-event images, labeled as changed and unchanged at pixel level, mainly caused by wildfire. [<a href="https://sites.google.com/site/michelevolpiresearch/codes/cross-sensor" target="_blank">Download</a>]</td>
    </tr>
	 <tr>
	    <td>Google data set [<a href="#Ref-23">23</a>]</td>
	    <td>19 season-varying VHR images pairswith 3 bands of red, green, and blue, a spatial resolution of 0.55 m, and the size ranging from 1006×1168 pixels to 4936×5224 pixels. The image changes include waters, roads, farmland, bare land, forests, buildings, ships, etc. Buildings make up the main changes. acquired during the periods between 2006 and 2019, covering the suburb areas of Guangzhou City, China. [<a href="https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery" target="_blank">Download</a>]</td>
    </tr>
    <tr>
	    <td rowspan="1" >Optical RS & SAR</td>
	    <td>California dataset [<a href="#Ref-22">22</a>]</td>
	    <td> 3 images, including a RS image captured by Landsat 8 with 9 channels on 2017, a SAR image captured by Sentinel-1A (recorded in polarisations VV and VH) after the occurrence of a flood, and a ground truth map. [<a href="https://sites.google.com/view/luppino/data" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td rowspan="3" >Street view</td>
	    <td>VL-CMU-CD  [<a href="#Ref-15">15</a>]</td>
	    <td>1362 co-registered pairs of RGB and depth images, labeled ground truth change (e.g., bin, sign, vehicle, refuse, construction, traffic cone, person/cycle, barrier) and sky masks at pixel level. [<a href="http://ghsi.github.io/proj/RSS2016.html" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>PCD 2015 [<a href="#Ref-16">16</a>]</td>
	    <td>200 panoramic image pairs in "TSUNAMI" and "GSV" subset, with the size of 224 × 1024 pixels, label as changed and unchanged at pixel level. [<a href="http://www.vision.is.tohoku.ac.jp/us/download/" target="_blank">Download</a>]</td>
	</tr>
	<tr>
	    <td>Change detection dataset  [<a href="#Ref-17">17</a>] </td>
	    <td>Image sequences of city streets captured by a vehicle-mounted camera at two different time points, with the size of 5000 × 2500 pixels, labeled 3D scene structure changes at pixel level. [<a href="http://www.vision.is.tohoku.ac.jp/us/research/4d_city_modeling/chg_dataset/" target="_blank">Download</a>]</td>
	</tr>
    	<tr>
       <td rowspan="3" >CV</td>
	    <td>CDNet 2012 [<a href="#Ref-18">18</a>] </td>
	    <td> 6 video categories with 4 to 6 videos sequences in each category, and the groundtruth images contain 5 labels namely: static, hard shadow, outside region of interest, unknown motion (usually around moving objects, due to semi-transparency and motion blur), and motion. [<a href="http://jacarini.dinf.usherbrooke.ca/dataset2012/" target="_blank">Download</a>]</td>
	</tr>
    	<tr>
	    <td>CDNet 2014  [<a href="#Ref-19">19</a>,<a href="#Ref-20">20</a>] </td>
	    <td> 22 additional videos (∼70; 000 pixel-wise annotated frames) spanning 5 new categories that incorporate challenges encountered in many surveillance settings, and provides realistic, camera captured (without CGI), diverse set of indoor and outdoor videos like the CDnet 2012. [<a href="http://www.changedetection.net/" target="_blank">Download</a>]
        </td>
	</tr>
    <tr>
	    <td colspan="2"> <a href="https://github.com/MinZHANG-WHU/Change-Detection-Review/blob/master/Video%20datasets.png" target="_blank"> More video datasets</a> </td>
	</tr>
</table>


It can be seen that the amount of open datasets that can be used for change detection tasks is small, and some of them have small data sizes. At present, there is still a lack of large SAR datasets that can be used for AI training. Most AI-based change detection methods are based on several SAR data sets that contain limited types of changes, e.g., the Bern dataset, the Ottawa dataset, the Yellow River dataset, and the Mexico dataset, which cannot meet the needs of change detection in areas with complex land cover and various change types. Moreover, their labels are not freely available. Street-view datasets are generally used for research of AI-based change detection methods in computer vision (CV). In CV, change detection based on pictures or video is also a hot research field, and the basic idea is consistent with that based on RS data. Therefore, in addition to street view image datasets, several video datasets in CV can also be used for research on AI-based change detection methods, such as CDNet 2012 and CDNet 2014. 

## 4. Applications
The development of AI-based change detection techniques has greatly facilitated many applications and has improved their automation and intelligence. Most AI-based change detection generates binary maps, and these studies only focus on the algorithm itself, without a specific application field. Therefore, it can be considered that they are generally suitable for LULC change detection. In this section, we focus on the techniques that are associated with specific applications, and they can be broadly divided into four categories:
* **Urban contexts**: urban expansion, public space management, and building change detection;
* **Resources and environment**: human-driven environmental changes, hydro-environmental changes, sea ice, surface water, and forest monitoring;
* **Natural disasters**: landslide mapping and damage assessment;
* **Astronomy**: planetary surfaces.

We provide an overview of the various change detection techniques in the literature for the different application categories. The works and data types associated with these applications are listed in Table 4.


<table>
<caption>Table 4. Summary of main applications of AI-based change detection techniques.</caption>
	<tr>
	    <th colspan="2">Applications</th>
	    <th>Data Types</th>
	    <th>Papers</th>  
	</tr>
	<tr>
	    <td rowspan="10">Urban contexts</td>
	    <td rowspan="2">Urban expansion</td>
	    <td>Satellite images  </td>
        <td><a href="https://dx.doi.org/10.3390/rs10030471" target="_blank">Lyu et.al (2018)</a>, <a href="https://dx.doi.org/10.1080/01431160903475290" target="_blank">Tong et.al (2007)</a></td>
	</tr>
    <tr>
	    <td>SAR images  </td>
        <td><a href="https://scholar.google.com/scholar_lookup?title=Generating+high-accuracy+urban+distribution+map+for+short-term+change+monitoring+based+on+convolutional+neural+network+by+utilizing+SAR+imagery&author=Iino,+S.&author=Ito,+R.&author=Doi,+K.&author=Imaizumi,+T.&author=Hikosaka,+S.&publication_year=2017" target="_blank">Iino et.al (2017)</a></td>
	</tr>
	<tr>
	    <td>Public space management</td>
	    <td>Street view images</td>
        <td><a href="https://scholar.google.com/scholar_lookup?title=ChangeNet:+A+deep+learning+architecture+for+visual+change+detection&conference=Proceedings+of+the+European+Conference+on+Computer+Vision+(ECCV)&author=Varghese,+A.&author=Gubbi,+J.&author=Ramaswamy,+A.&author=Balamuralidhar,+P.&publication_year=2018&pages=129%E2%80%93145" target="_blank">Varghese et.al (2018)</a></td>
	</tr>
    <tr>
	    <td>Road surface</td>
	    <td>UAV images</td>
        <td><a href="https://doi.org/10.3390/su12062482" target="_blank">Truong et.al (2020)</a></td>
	</tr>
    <tr>
	    <td rowspan="6">Building change detection</td>
	    <td>Aerial images</td>
        <td><a href="https://dx.doi.org/10.3390/rs11111343" target="_blank">Ji et.al (2019)</a>, <a href="https://scholar.google.com/scholar_lookup?title=A+deep+learning+approach+to+detecting+changes+in+buildings+from+aerial+images&conference=Proceedings+of+the+International+Symposium+on+Neural+Networks&author=Sun,+B.&author=Li,+G.-Z.&author=Han,+M.&author=Lin,+Q.-H.&publication_year=2019&pages=414%E2%80%93421" target="_blank">Sun et.al (2019)</a>, <a href="https://dx.doi.org/10.1117/12.2277912" target="_blank">Nemoto et.al (2017)</a></td>
	</tr>
    <tr>
	    <td>Satellite images</td>
        <td><a href="https://dx.doi.org/10.1016/j.jvcir.2019.102585" target="_blank">Huang et.al (2019)</a>, <a href="https://scholar.google.com/scholar_lookup?title=Change+Detection+Based+on+the+Combination+of+Improved+SegNet+Neural+Network+and+Morphology&conference=Proceedings+of+the+2018+IEEE+3rd+International+Conference+on+Image,+Vision+and+Computing+(ICIVC)&author=Zhu,+B.&author=Gao,+H.&author=Wang,+X.&author=Xu,+M.&author=Zhu,+X.&publication_year=2018&pages=55%E2%80%9359" target="_blank">Zhu et.al (2018)</a></td>
	</tr>
    <tr>
	    <td>Satellite/Aerial images</td>
        <td><a href="https://dx.doi.org/10.3390/rs12030484" target="_blank">Jiang  et.al (2020)</a>, <a href="https://dx.doi.org/10.1109/TGRS.2018.2858817" target="_blank">Ji et.al (2018)</a>, <a href="https://dx.doi.org/10.1109/TGRS.2020.3000296" target="_blank">Saha et.al (2020)</a></td>
	</tr>
    <tr>
	    <td>Airborne laser scanning data and aerial images </td>
        <td><a href="https://dx.doi.org/10.3390/rs11202417" target="_blank">Zhang et.al (2019)</a></td>
	</tr>
    <tr>
	    <td>SAR images </td>
        <td><a href="https://dx.doi.org/10.3390/rs11121444" target="_blank">Jaturapitpornchai et.al (2019)</a></td>
	</tr>
    <tr>
	    <td>Satellite images and GIS map</td>
        <td><a href="https://dx.doi.org/10.3390/rs11202427" target="_blank">Ghaffarian et.al (2019)</a></td>
	</tr>
    <tr>
	    <td rowspan="5">Resources & environment </td>
	    <td>Human-driven environmental changes</td>
	    <td>Satellite images  </td>
        <td><a href="https://dx.doi.org/10.1117/1.JRS.10.016021" target="_blank">Chen et.al (2016)</a></td>
	</tr>
    <tr>
	    <td>Hydro-environmental changes</td>
	    <td>Satellite images</td>
        <td><a href="https://dx.doi.org/10.1016/j.jhydrol.2018.05.018" target="_blank">Nourani et.al (2018)</a></td>
	</tr>
     <tr>
	    <td>Sea ice</td>
	    <td>SAR images</td>
        <td><a href="https://dx.doi.org/10.1109/LGRS.2019.2906279" target="_blank">Gao et.al (2019)</a>, <a href="https://dx.doi.org/10.1109/LGRS.2019.2895656" target="_blank">Gao et.al (2019)</a></td>
	</tr>
    <tr>
	    <td>Surface water</td>
	    <td>Satellite images</td>
        <td><a href="https://dx.doi.org/10.2112/SI91-086.1" target="_blank">Song et.al (2019)</a>, <a href="https://dx.doi.org/10.1016/j.jag.2014.08.014" target="_blank">Rokni et.al (2015)</a></td>
	</tr>
    <tr>
	    <td>Forest monitoring</td>
	    <td>Satellite images</td>
        <td><a href="https://dx.doi.org/10.1109/TGRS.2017.2707528" target="_blank">Khan et.al (2017)</a>, <a href="https://dx.doi.org/10.3390/rs8080678" target="_blank">Lindquist et.al (2016)</a>, <a href="https://scholar.google.com/scholar_lookup?title=Comparison+of+pixel+-based+and+artificial+neural+networks+classification+methods+for+detecting+forest+cover+changes+in+Malaysia&conference=Proceedings+of+the+8th+International+Symposium+of+the+Digital+Earth,+Univ+Teknologi+Malaysia,+Inst+Geospatial+Sci+&+Technol&author=Deilmai,+B.R.&author=Kanniah,+K.D.&author=Rasib,+A.W.&author=Ariffin,+A.&publication_year=2014" target="_blank">Deilmai et.al (2014)</a>, <a href="https://dx.doi.org/10.1016/S0034-4257(01)00259-0" target="_blank">Woodcock et.al (2001)</a>, <a href="https://dx.doi.org/10.1109/36.485117" target="_blank">Gopal et.al (1996)</a></td>
	</tr>
    <tr>
	    <td rowspan="7">Natural disasters</td>
	    <td rowspan="2">Landslide mapping</td>
	    <td>Aerial images</td>
        <td><a href="https://dx.doi.org/10.1109/LGRS.2020.2979693" target="_blank">Fang et.al (2020)</a>, <a href="https://dx.doi.org/10.1109/LGRS.2018.2889307" target="_blank">Lei et.al (2019)</a></td>
	</tr>
    <tr>
	    <td>Satellite images</td>
        <td><a href="https://dx.doi.org/10.3390/s18030821" target="_blank">Chen et.al (2018)</a>, <a href="https://scholar.google.com/scholar_lookup?title=Automatic+Recognition+of+Landslide+Based+on+CNN+and+Texture+Change+Detection&conference=Proceedings+of+the+2016+31st+Youth+Academic+Annual+Conference+of+Chinese-Association-of-Automation+(YAC)&author=Ding,+A.&author=Zhang,+Q.&author=Zhou,+X.&author=Dai,+B.&publication_year=2016&pages=444%E2%80%93448" target="_blank">Ding et.al (2016)</a>, <a href="https://dx.doi.org/10.1007/s11069-006-9041-x" target="_blank">Tarantino et.al (2006)</a></td>
	</tr>
    <tr>
	    <td rowspan="5">Damage assessment </td>
	    <td>Satellite images</td>
        <td>caused by tsunami [<a href="https://dx.doi.org/10.3390/rs11091123" target="_blank">Sublime et.al (2019)</a>,<a href="https://dx.doi.org/10.1007/s11069-015-1595-z" target="_blank">Singh et.al (2015)</a>], particular incident [<a href="https://scholar.google.com/scholar_lookup?title=Change+detection+from+unlabeled+remote+sensing+images+using+siamese+ANN&conference=Proceedings+of+the+IGARSS+2019%E2%80%942019+IEEE+International+Geoscience+and+Remote+Sensing+Symposium&author=Hedjam,+R.&author=Abdesselam,+A.&author=Melgani,+F.&publication_year=2019&pages=1530%E2%80%931533" target="_blank">Hedjam et.al (2019)</a>], flood [<a href="https://dx.doi.org/10.3390/rs11212492" target="_blank">Peng et.al (2019)</a>], or earthquake [<a href="https://dx.doi.org/10.3390/rs11101202" target="_blank">Ji et.al (2019)</a>]</td>
	</tr>
    <tr>
	    <td>Aerial images</td>
        <td>caused by tsunami [<a href="https://scholar.google.com/scholar_lookup?title=Damage+detection+from+aerial+images+via+convolutional+neural+networks&conference=Proceedings+of+the+2017+Fifteenth+IAPR+International+Conference+on+Machine+Vision+Applications+(MVA),+Nagoya+Univ&author=Fujita,+A.&author=Sakurada,+K.&author=Imaizumi,+T.&author=Ito,+R.&author=Hikosaka,+S.&author=Nakamura,+R.&publication_year=2017&pages=5%E2%80%938" target="_blank">Fujita et.al (2017)</a>]</td>
	</tr>
    <tr>
	    <td>SAR images</td>
        <td>caused by fires  [<a href="https://dx.doi.org/10.1109/LGRS.2017.2786344" target="_blank">Planinšič et.al (2018)</a>], or earthquake [<a href="https://scholar.google.com/scholar_lookup?title=Destroyed-buildings+detection+from+VHR+SAR+images+using+deep+features&author=Saha,+S.&author=Bovolo,+F.&author=Bruzzone,+L.&publication_year=2018" target="_blank">Saha et.al (2018)</a>]</td>
	</tr>
    <tr>
	    <td>Street view images </td>
        <td>caused by tsunami [<a href="https://scholar.google.com/scholar_lookup?title=Change+detection+from+a+street+image+pair+using+CNN+features+and+superpixel+segmentation&conference=Proceedings+of+the+British+Machine+Vision+Conference+(BMVC)&author=Sakurada,+K.&author=Okatani,+T.&publication_year=2015&pages=61.1%E2%80%9361.12" target="_blank">Sakurada et.al (2015)</a>]</td>
	</tr>
    <tr>
	    <td>Street view images and GIS map </td>
        <td>caused by tsunami [<a href="https://dx.doi.org/10.1016/j.cviu.2017.01.012" target="_blank">Sakurada et.al (2017)</a>]</td>
	</tr>
    <tr>
	    <td>Astronomy</td>
	    <td>Planetary surfaces</td>
	    <td>Satellite images</td>
        <td><a href="https://dx.doi.org/10.1109/JSTARS.2019.2936771" target="_blank">Kerner et.al (2019)</a></td>
	</tr>
</table>

## 5. Software programs
There are currently a large number of software with change detection tools, and we have a brief summary of them, see table 5.
<table>
<caption>Table 5. A list of software for change detection.</caption>
	<tr>
	    <th>Type</th>
	    <th>Name</th>
        <th>Description</th>
	</tr>
    <tr>
    <td rowspan="6">Commercial</td>
    <td>ERDAS IMAGINE</td>
    <td>provides true value, consolidating remote sensing, photogrammetry, LiDAR analysis, basic vector analysis, and radar processing into a single product, including a variety of <a href="https://www.hexagongeospatial.com/products/power-portfolio/erdas-imagine/erdas-imagine-remote-sensing-software-package" target="_blank">change detection tools</a>.</td>
    </tr>
    <tr>
    <td>ArcGIS</td>
    <td> change detection can be calculate between two raster datasets by using the <a href="https://support.esri.com/en/technical-article/000001209" target="_blank">raster calculator tool</a> or <a href="https://pro.arcgis.com/en/pro-app/help/analysis/image-analyst/deep-learning-in-arcgis-pro.htm" target="_blank">deep learning workflow</a>. </td>
    </tr>
     <tr>
    <td>ENVI</td>
    <td>provides <a href="https://www.harrisgeospatial.com/docs/ChangeDetectionAnalysis.html" target="_blank">change detection analysis tools</a> and the <a href="https://www.harrisgeospatial.com/Software-Technology/ENVI-Deep-Learning" target="_blank"> ENVI deep learning module</a>.</td>
    </tr>
     <tr>
    <td>eCognition</td>
    <td>can be used for <a href="https://geospatial.trimble.com/products-and-solutions/ecog-essentials-support-cases" target="_blank">a variety of change mapping</a>, and by leveraging deep learning technology from the Google TensorFlow™ library, eCognition empowers customers with highly sophisticated pattern recognition and correlation tools that automate the classification of objects of interest for faster and more accurate results,<a href="https://geospatial.trimble.com/ecognition-whats-new" target="_blank"> more</a>.</td>
    </tr>
     <tr>
    <td>PCI Geomatica</td>
    <td> provides <a href="https://support.pcigeomatics.com/hc/en-us/articles/203483499-Change-Detection-Optical" target="_blank">change detection tools</a>, and can be useful in numerous circumstances in which you may want to analyze change, such as: storm damage, forest-fire damage, flooding, urban sprawl, and <a href="https://support.pcigeomatics.com/hc/en-us/articles/203483499-Change-Detection-Optical" target="_blank">more</a>.</td>
    </tr>
    <tr>
    <td>SenseTime</td>
    <td> <a href="https://www.sensetime.com/en/Service/RemoteSensing.html#product" target="_blank">SenseRemote remote sensing intelligent solutions</a></td>
    </tr>
    <tr>
    <td rowspan="3">Open source</td>
    <td>QGIS</td>
    <td>provides many <a href="https://plugins.qgis.org/plugins/tags/change-detection/" target="_blank">change detection tools</a>.</td>
    </tr>
     <tr>
    <td>Orfeo ToolBox</td>
    <td>change detection by <a href="https://www.orfeo-toolbox.org/CookBook/Applications/Change_Detection.html" target="_blank">multivariate alteration detector (MAD) algorithm</a>.</td>
    </tr>
   <tr>
    <td>Change Detection ToolBox</td>
    <td><a href="https://github.com/Bobholamovic/ChangeDetectionToolbox" target="_blank">MATLAB toolbox for remote sensing change detection</a>.</td>
    </tr>
<table>

## 6. Review papers for change detection
The following papers are helpful for researchers to better understand this  field of remote sensing change detection, see table 6.
<table>
<caption>Table 6. A list of review papers on change detection.</caption>
	<tr>
	    <th>Published year</th>
	    <th>Review paper</th>
	</tr>
    <tr>
    <td>1989</td>
    <td>Digital change detection techniques using remotely sensed data, IJRS. [<a href="https://dx.doi.org/10.1080/01431168908903939" target="_blank">paper</a>]</td>
    </tr>
	 <tr>
    <td>2004</td>
    <td>Digital change detection methods in ecosystem monitoring: a review, IJRS. [<a href="https://dx.doi.org/10.1080/0143116031000101675" target="_blank">paper</a>]</td>
    </tr>
	 <tr>
	 <td>2004</td>
    <td>Change detection techniques, IJRS. [<a href="https://dx.doi.org/10.1080/0143116031000139863" target="_blank">paper</a>]</td>
    </tr>
	 <tr>
	<td>2012</td>
    <td>Object-based change detection, IJRS. [<a href="https://dx.doi.org/10.1080/01431161.2011.648285" target="_blank">paper</a>]</td>
    </tr>
	 <tr>
	 <td>2013</td>
    <td>Change detection from remotely sensed images: From pixel-based to object-based approaches, ISPRS. [<a href="https://doi.org/10.1016/j.isprsjprs.2013.03.006" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2016</td>
    <td>3D change detection–approaches and applications, ISPRS. [<a href="https://doi.org/10.1016/j.isprsjprs.2016.09.013" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2016</td>
    <td>Deep learning for remote sensing data a technical tutorial on the state of the art, MGRS. [<a href="https://dx.doi.org/10.1109/MGRS.2016.2540798" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2017</td>
    <td>Comprehensive survey of deep learning in remote sensing: theories, tools, and challenges for the community, JRS. [<a href="https://doi.org/10.1117/1.JRS.11.042609" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2017</td>
    <td>Deep Learning in Remote Sensing, MGRS. [<a href="https://dx.doi.org/10.1109/MGRS.2017.2762307" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2018</td>
    <td>Computational intelligence in optical remote sensing image processing, ASOC. [<a href="https://doi.org/10.1016/j.asoc.2017.11.045" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2019</td>
    <td>A review of change detection in multitemporal hyperspectral images: current techniques, applications, and challenges, MGRS. [<a href="https://dx.doi.org/10.1109/MGRS.2019.2898520" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2019</td>
    <td>Deep learning in remote sensing applications: A meta-analysis and review, ISPRS. [<a href="https://doi.org/10.1016/j.isprsjprs.2019.04.015" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2020</td>
    <td>Deep Learning for change detection in remote sensing images: comprehensive review and meta-analysis, arXiv. [<a href="https://arxiv.org/abs/2006.05612" target="_blank">paper</a>]</td>
    </tr>
	<tr>
	 <td>2020</td>
    <td>Change detection based on artificial intelligence: state-of-the-art and challenges, RS. [<a href="https://doi.org/10.3390/rs12101688" target="_blank">paper</a>]</td>
    </tr>
<table>

## 7. Reference
<span id="Ref-1">[1] Hyperspectral Change Detection Dataset. Available online: https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset (accessed on 4 May 2020).</span>

<span id="Ref-2">[2] Wang, Q.; Yuan, Z.; Du, Q.; Li, X. GETNET: A General End-to-End 2-D CNN Framework for Hyperspectral Image Change Detection. IEEE Trans. Geosci. Remote Sens. 2018, 57, 3–13. [<a href="https://scholar.google.com/scholar_lookup?title=GETNET:+A+General+End-to-End+2-D+CNN+Framework+for+Hyperspectral+Image+Change+Detection&author=Wang,+Q.&author=Yuan,+Z.&author=Du,+Q.&author=Li,+X.&publication_year=2018&journal=IEEE+Trans.+Geosci.+Remote+Sens.&volume=57&pages=3%E2%80%9313&doi=10.1109/TGRS.2018.2849692" target="_blank">Google Scholar</a>] [<a href="https://ieeexplore.ieee.org/document/8418840/" target="_blank">CrossRef</a>]</span>

<span id="Ref-3">[3] Daudt, R.C.; Le Saux, B.; Boulch, A.; Gousseau, Y. Multitask learning for large-scale semantic change detection. Comput. Vis. Image Underst. 2019, 187, 102783. [<a href="https://scholar.google.com/scholar_lookup?title=Multitask+learning+for+large-scale+semantic+change+detection&author=Daudt,+R.C.&author=Le+Saux,+B.&author=Boulch,+A.&author=Gousseau,+Y.&publication_year=2019&journal=Comput.+Vis.+Image+Underst.&volume=187&pages=102783&doi=10.1016/j.cviu.2019.07.003" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.1016/j.cviu.2019.07.003" target="_blank">CrossRef</a>]</span>

<span id="Ref-4">[4] Ji, S.; Wei, S.; Lu, M. Fully Convolutional Networks for Multisource Building Extraction from an Open Aerial and Satellite Imagery Data Set. IEEE Trans. Geosci. Remote Sens. 2018, 57, 574–586. [<a href="https://scholar.google.com/scholar_lookup?title=Fully+Convolutional+Networks+for+Multisource+Building+Extraction+from+an+Open+Aerial+and+Satellite+Imagery+Data+Set&author=Ji,+S.&author=Wei,+S.&author=Lu,+M.&publication_year=2018&journal=IEEE+Trans.+Geosci.+Remote+Sens.&volume=57&pages=574%E2%80%93586&doi=10.1109/TGRS.2018.2858817" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.1109/TGRS.2018.2858817" target="_blank">CrossRef</a>]</span>

<span id="Ref-5">[5] Benedek, C.; Sziranyi, T. Change Detection in Optical Aerial Images by a Multilayer Conditional Mixed Markov Model. IEEE Trans. Geosci. Remote Sens. 2009, 47, 3416–3430. [<a href="https://scholar.google.com/scholar_lookup?title=Change+Detection+in+Optical+Aerial+Images+by+a+Multilayer+Conditional+Mixed+Markov+Model&author=Benedek,+C.&author=Sziranyi,+T.&publication_year=2009&journal=IEEE+Trans.+Geosci.+Remote+Sens.&volume=47&pages=3416%E2%80%933430&doi=10.1109/TGRS.2009.2022633" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.1109/TGRS.2009.2022633" target="_blank">CrossRef</a>]</span>

<span id="Ref-6">[6] Benedek, C.; Sziranyi, T. A Mixed Markov model for change detection in aerial photos with large time differences. In Proceedings of the 2008 19th International Conference on Pattern Recognition, Tampa, FL, USA, 8–11 December 2008; pp. 1–4. [<a href="https://scholar.google.com/scholar_lookup?title=A+Mixed+Markov+model+for+change+detection+in+aerial+photos+with+large+time+differences&conference=Proceedings+of+the+2008+19th+International+Conference+on+Pattern+Recognition&author=Benedek,+C.&author=Sziranyi,+T.&publication_year=2008&pages=1%E2%80%934" target="_blank">Google Scholar</a>] </span>

<span id="Ref-7">[7] Daudt, R.C.; Le Saux, B.; Boulch, A.; Gousseau, Y. Urban change detection for multispectral earth observation using convolutional neural networks. In Proceedings of the IGARSS 2018 IEEE International Geoscience and Remote Sensing Symposium, Valencia, Spain, 22–27 July 2018; pp. 2115–2118. [<a href="https://scholar.google.com/scholar_lookup?title=Urban+change+detection+for+multispectral+earth+observation+using+convolutional+neural+networks&conference=Proceedings+of+the+IGARSS+2018+IEEE+International+Geoscience+and+Remote+Sensing+Symposium&author=Daudt,+R.C.&author=Le+Saux,+B.&author=Boulch,+A.&author=Gousseau,+Y.&publication_year=2018&pages=2115%E2%80%932118" target="_blank">Google Scholar</a>]</span>

<span id="Ref-8">[8] Zhang, M.; Shi, W. A Feature Difference Convolutional Neural Network-Based Change Detection Method. IEEE Trans. Geosci. Remote Sens. 2020, 1–15.  [<a href="https://scholar.google.com/scholar_lookup?title=A+Feature+Difference+Convolutional+Neural+Network-Based+Change+Detection+Method&author=Zhang,+M.&author=Shi,+W.&publication_year=2020&journal=IEEE+Trans.+Geosci.+Remote+Sens.&pages=1%E2%80%9315&doi=10.1109/tgrs.2020.2981051" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.1109/tgrs.2020.2981051" target="_blank">CrossRef</a>]</span>

<span id="Ref-9">[9] Wu, C.; Zhang, L.; Zhang, L. A scene change detection framework for multi-temporal very high resolution remote sensing images. Signal Process. 2016, 124, 184–197. [<a href="https://scholar.google.com/scholar_lookup?title=A+scene+change+detection+framework+for+multi-temporal+very+high+resolution+remote+sensing+images&author=Wu,+C.&author=Zhang,+L.&author=Zhang,+L.&publication_year=2016&journal=Signal+Process.&volume=124&pages=184%E2%80%93197&doi=10.1016/j.sigpro.2015.09.020" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.1016/j.sigpro.2015.09.020" target="_blank">CrossRef</a>]</span>

<span id="Ref-10">[10] Fujita, A.; Sakurada, K.; Imaizumi, T.; Ito, R.; Hikosaka, S.; Nakamura, R. Damage detection from aerial images via convolutional neural networks. In Proceedings of the 2017 Fifteenth IAPR International Conference on Machine Vision Applications (MVA), Nagoya Univ, Nagoya, Japan, 08–12 May 2017; pp. 5–8 [<a href="https://scholar.google.com/scholar_lookup?title=Damage+detection+from+aerial+images+via+convolutional+neural+networks&conference=Proceedings+of+the+2017+Fifteenth+IAPR+International+Conference+on+Machine+Vision+Applications+(MVA),+Nagoya+Univ&author=Fujita,+A.&author=Sakurada,+K.&author=Imaizumi,+T.&author=Ito,+R.&author=Hikosaka,+S.&author=Nakamura,+R.&publication_year=2017&pages=5%E2%80%938" target="_blank">Google Scholar</a>] </span>

<span id="Ref-11">[11] Gupta, R.; Goodman, B.; Patel, N.; Hosfelt, R.; Sajeev, S.; Heim, E.; Doshi, J.; Lucas, K.; Choset, H.; Gaston, M. Creating xBD: A dataset for assessing building damage from satellite imagery. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, Long Beach, CA, USA, 16–20 June 2019; pp. 10–17. [<a href="https://scholar.google.com/scholar_lookup?title=Creating+xBD:+A+dataset+for+assessing+building+damage+from+satellite+imagery&conference=Proceedings+of+the+IEEE+Conference+on+Computer+Vision+and+Pattern+Recognition+Workshops&author=Gupta,+R.&author=Goodman,+B.&author=Patel,+N.&author=Hosfelt,+R.&author=Sajeev,+S.&author=Heim,+E.&author=Doshi,+J.&author=Lucas,+K.&author=Choset,+H.&author=Gaston,+M.&publication_year=2019&pages=10%E2%80%9317" target="_blank">Google Scholar</a>]</span>

<span id="Ref-12">[12] Bourdis, N.; Marraud, D.; Sahbi, H. Constrained optical flow for aerial image change detection. In Proceedings of the 2011 IEEE International Geoscience and Remote Sensing Symposium, Vancouver, BC, Canada, 24–29 July 2011; pp. 4176–4179. [<a href="https://scholar.google.com/scholar_lookup?title=Constrained+optical+flow+for+aerial+image+change+detection&conference=Proceedings+of+the+2011+IEEE+International+Geoscience+and+Remote+Sensing+Symposium&author=Bourdis,+N.&author=Marraud,+D.&author=Sahbi,+H.&publication_year=2011&pages=4176%E2%80%934179&doi=10.1109/igarss.2011.6050150" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.1109/igarss.2011.6050150" target="_blank">CrossRef</a>]</span>

<span id="Ref-13">[13] Lebedev, M.A.; Vizilter, Y.V.; Vygolov, O.V.; Knyaz, V.A.; Rubis, A.Y. Change detection in remote sensing images using conditional adversarial networks. ISPRS Int. Arch. Photogramm. Remote Sens. Spat. Inf. Sci. 2018, 565–571. [<a href="https://scholar.google.com/scholar_lookup?title=Change+detection+in+remote+sensing+images+using+conditional+adversarial+networks&author=Lebedev,+M.A.&author=Vizilter,+Y.V.&author=Vygolov,+O.V.&author=Knyaz,+V.A.&author=Rubis,+A.Y.&publication_year=2018&journal=ISPRS+Int.+Arch.+Photogramm.+Remote+Sens.+Spat.+Inf.+Sci.&pages=565%E2%80%93571&doi=10.5194/isprs-archives-XLII-2-565-2018" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.5194/isprs-archives-XLII-2-565-2018" target="_blank">CrossRef</a>]</span>

<span id="Ref-14">[14] Chen, H.; Shi, Z. A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection. Remote Sensing, 12(10), 1662. [<a href="https://scholar.google.co.jp/scholar?q=A+Spatial-Temporal+Attention-Based+Method+and+a+New+Dataset+for+Remote+Sensing+Image+Change+Detection&hl=zh-TW&as_sdt=0&as_vis=1&oi=scholart" target="_blank">Google Scholar</a>] [<a href="https://doi.org/10.3390/rs12101662" target="_blank">CrossRef</a>]</span>

<span id="Ref-15">[15] Alcantarilla, P.F.; Stent, S.; Ros, G.; Arroyo, R.; Gherardi, R. Street-view change detection with deconvolutional networks. Auton. Robot. 2018, 42, 1301–1322. [<a href="https://scholar.google.com/scholar_lookup?title=Street-view+change+detection+with+deconvolutional+networks&author=Alcantarilla,+P.F.&author=Stent,+S.&author=Ros,+G.&author=Arroyo,+R.&author=Gherardi,+R.&publication_year=2018&journal=Auton.+Robot.&volume=42&pages=1301%E2%80%931322&doi=10.1007/s10514-018-9734-5" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.1007/s10514-018-9734-5" target="_blank">CrossRef</a>]</span>

<span id="Ref-16">[16] Sakurada, K.; Okatani, T. Change detection from a street image pair using CNN features and superpixel segmentation. In Proceedings of the British Machine Vision Conference (BMVC), Swansea, UK, 7–10 September 2015; pp. 61.1–61.12. [<a href="https://scholar.google.com/scholar_lookup?title=Change+detection+from+a+street+image+pair+using+CNN+features+and+superpixel+segmentation&conference=Proceedings+of+the+British+Machine+Vision+Conference+(BMVC)&author=Sakurada,+K.&author=Okatani,+T.&publication_year=2015&pages=61.1%E2%80%9361.12" target="_blank">Google Scholar</a>] </span>

<span id="Ref-17">[17] Sakurada, K.; Okatani, T.; Deguchi, K. Detecting changes in 3D structure of a scene from multi-view images captured by a vehicle-mounted camera. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Portland, OR, USA, 23–28 June 2013; pp. 137–144. [<a href="https://scholar.google.com/scholar_lookup?title=Detecting+changes+in+3D+structure+of+a+scene+from+multi-view+images+captured+by+a+vehicle-mounted+camera&conference=Proceedings+of+the+IEEE+Conference+on+Computer+Vision+and+Pattern+Recognition&author=Sakurada,+K.&author=Okatani,+T.&author=Deguchi,+K.&publication_year=2013&pages=137%E2%80%93144" target="_blank">Google Scholar</a>]</span>

<span id="Ref-18">[18] Goyette, N.; Jodoin, P.-M.; Porikli, F.; Konrad, J.; Ishwar, P. Changedetection. net: A new change detection benchmark dataset. In Proceedings of the 2012 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, Providence, RI, USA, 16–21 June 2012; pp. 1–8. [<a href="https://scholar.google.com/scholar_lookup?title=Changedetection.+net:+A+new+change+detection+benchmark+dataset&conference=Proceedings+of+the+2012+IEEE+Computer+Society+Conference+on+Computer+Vision+and+Pattern+Recognition+Workshops&author=Goyette,+N.&author=Jodoin,+P.-M.&author=Porikli,+F.&author=Konrad,+J.&author=Ishwar,+P.&publication_year=2012&pages=1%E2%80%938" target="_blank">Google Scholar</a>] </span>

<span id="Ref-19">[19] Wang, Y.; Jodoin, P.-M.; Porikli, F.; Konrad, J.; Benezeth, Y.; Ishwar, P. CDnet 2014: An Expanded Change Detection Benchmark Dataset. In Proceedings of the 2014 IEEE Conference on Computer Vision and Pattern Recognition Workshops, Columbus, OH, USA, 23–28 June 2014; pp. 393–400.  [<a href="https://scholar.google.com/scholar_lookup?title=CDnet+2014:+An+Expanded+Change+Detection+Benchmark+Dataset&conference=Proceedings+of+the+2014+IEEE+Conference+on+Computer+Vision+and+Pattern+Recognition+Workshops&author=Wang,+Y.&author=Jodoin,+P.-M.&author=Porikli,+F.&author=Konrad,+J.&author=Benezeth,+Y.&author=Ishwar,+P.&publication_year=2014&pages=393%E2%80%93400" target="_blank">Google Scholar</a>] </span>

<span id="Ref-20">[20] Goyette, N.; Jodoin, P.-M.; Porikli, F.; Konrad, J.; Ishwar, P. A Novel Video Dataset for Change Detection Benchmarking. IEEE Trans. Image Process. 2014, 23, 4663–4679. [<a href="https://scholar.google.com/scholar_lookup?title=A+Novel+Video+Dataset+for+Change+Detection+Benchmarking&author=Goyette,+N.&author=Jodoin,+P.-M.&author=Porikli,+F.&author=Konrad,+J.&author=Ishwar,+P.&publication_year=2014&journal=IEEE+Trans.+Image+Process.&volume=23&pages=4663%E2%80%934679&doi=10.1109/TIP.2014.2346013" target="_blank">Google Scholar</a>] [<a href="https://dx.doi.org/10.1109/TIP.2014.2346013" target="_blank">CrossRef</a>] </span>

<span id="Ref-21">[21] Volpi, Michele; Camps-Valls, Gustau; Tuia, Devis (2015). Spectral alignment of multi-temporal cross-sensor images with automated kernel canonical correlation analysis; ISPRS Journal of Photogrammetry and Remote Sensing, vol. 107, pp. 50-63, 2015. [<a href="https://dx.doi.org/10.1016/j.isprsjprs.2015.02.005" target="_blank">CrossRef</a>] </span>

<span id="Ref-22">[22] L. T. Luppino, F. M. Bianchi, G. Moser and S. N. Anfinsen. Unsupervised Image Regression for Heterogeneous Change Detection. IEEE Transactions on Geoscience and Remote Sensing. 2019, vol. 57, no. 12, pp. 9960-9975. [<a href="https://dx.doi.org/10.1109/TGRS.2019.2930348" target="_blank">CrossRef</a>] </span>

<span id="Ref-23">[23] D. Peng, L. Bruzzone, Y. Zhang, H. Guan, H. Ding and X. Huang, SemiCDNet: A Semisupervised Convolutional Neural Network for Change Detection in High Resolution Remote-Sensing Images. IEEE Transactions on Geoscience and Remote Sensing. 2020. [<a href="https://dx.doi.org/10.1109/TGRS.2020.3011913" target="_blank">CrossRef</a>] </span>

<span id="Ref-24">[24] Yang, Kunping, et al. Asymmetric Siamese Networks for Semantic Change Detection. arXiv preprint arXiv:2010.05687 (2020). [<a href="https://arxiv.org/abs/2010.05687" target="_blank">CrossRef</a>] </span>

## Cite
If you find this review helpful to you, please consider citing our paper. [<a href="https://doi.org/10.3390/rs12101688" target="_blank">Open Access</a>]

```
@Article{rs12101688,
AUTHOR = {Shi, Wenzhong and Zhang, Min and Zhang, Rui and Chen, Shanxiong and Zhan, Zhao},
TITLE = {Change Detection Based on Artificial Intelligence: State-of-the-Art and Challenges},
JOURNAL = {Remote Sensing},
VOLUME = {12},
YEAR = {2020},
NUMBER = {10},
ARTICLE-NUMBER = {1688},
URL = {https://www.mdpi.com/2072-4292/12/10/1688},
ISSN = {2072-4292},
DOI = {10.3390/rs12101688}
}
```

## Note
This list will be updated in time, and volunteer contributions are welcome. For questions or sharing, please feel free to [contact us](mailto:007zhangmin@whu.edu.cn) or make issues.

##### Reference materials:
* [Michele Volpi personal research page](https://sites.google.com/site/michelevolpiresearch/codes)
* [llu025/Heterogeneous_CD](https://github.com/llu025/Heterogeneous_CD)
* [wenhwu/awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection)
* [neverstoplearn/remote_sensing_change_detection](https://github.com/neverstoplearn/remote_sensing_change_detection)
* [Change Detection in GIS](https://www.gislounge.com/change-detection-in-gis/)
* [Gao Feng personal research page](http://feng-gao.cn/)
* [Bobholamovic/ChangeDetectionToolbox](https://github.com/Bobholamovic/ChangeDetectionToolbox)

