# Change Detection Based on Artificial Intelligence: State-of-the-Art and Challenges

## Introduction
 Change detection based on remote sensing (RS) data is an important method of detecting changes on the Earth’s surface and has a wide range of applications in urban planning, environmental monitoring, agriculture investigation, disaster assessment, and map revision. In recent years, integrated artificial intelligence (AI) technology has become a research focus in developing new change detection methods. Although some researchers claim that AI-based change detection approaches outperform traditional change detection approaches, it is not immediately obvious how and to what extent AI can improve the performance of change detection. This review focuses on the state-of-the-art methods, applications, and challenges of AI for change detection. Specifically, the implementation process of AI-based change detection is first introduced. Then, the data from different sensors used for change detection, including optical RS data, synthetic aperture radar (SAR) data, street view images, and combined heterogeneous data, are presented, and the available open datasets are also listed. The general frameworks of AI-based change detection methods are reviewed and analyzed systematically, and the unsupervised schemes used in AI-based change detection are further analyzed. Subsequently, the commonly used networks in AI for change detection are described. From a practical point of view, the application domains of AI-based change detection methods are classified based on their applicability. Finally, the major challenges and prospects of AI for change detection are discussed and delineated, including (a) heterogeneous big data processing, (b) unsupervised AI, and (c) the reliability of AI. This review will be beneficial for researchers in understanding this field.
 
![](/Figure%201.png)
Figure 1. General schematic diagram of change detection.

<br/>

![](/Figure%202.png)
Figure 2. Implementation process of AI-based change detection (black arrows indicate workflow and red arrow indicates an example).

## Implementation process

Figure 2 provide a general implementation process of AI-based change detection, but the structure of the AI model is diverse and needs to be well designed according to different application situations and the training data. It is worth mentioning that existing mature frameworks such as TensorFlow, Keras, Pytorch, and Caffe, help researchers more easily realize the design, training, and deployment of AI models, and their development documents provide detailed introductions.

TODO

## Open datasets

Currently, there are some freely available data sets for change detection, which can be used as benchmark datasets for AI training and accuracy evaluation in future research. Detailed information is presented in Table 1.

<table>
<caption>Tabel 1. A list of open datasets for change detection.<caption>
	<tr>
	    <th>Type</th>
	    <th width="180px">Data set</th>
	    <th>Description</th>  
	</tr>
	<tr>
	    <td rowspan="13">Optical RS</td>
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


## Reference
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
This list will be updated in time, and volunteer contributions are welcome.
