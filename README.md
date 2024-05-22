# Reconstructing-BP-waves-from-iPPG-signals
Reconstructing blood pressure waves from imaging photoplethysmographic signals
In this study, we propose to convert photoplethysmographic signals(iPPG) 
into blood pressure(BP) signals using their continuous wavelet transforms (CWTs).
The real and imaginary parts of the CWT are passed to a pre-trained deep architecture
(ResNeXt101) called U -shaped, trained using the BP4D+ dataset.

# Dataset
For the study was used a subset of the BP4D+ dataset. 
The “BP4D+”, extended from the BP4D database, is a Multimodal Spontaneous Emotion Corpus (MMSE), 
which contains multimodal datasets including synchronized 3D, 2D, thermal, physiological data sequences (e.g., heart rate, blood pressure, skin conductance (EDA), 
and respiration rate), and meta-data (facial features and FACS codes). 

There are 140 subjects, including 58 males and 82 females, with ages ranging from 18 to 66 years old. 
Ethnic/Racial Ancestries include Black, White, Asian (including East-Asian and Middle-East-Asian), Hispanic/Latino, and others (e.g., Native American). 

With 140 subjects and 10 tasks (emotions) for each subject included in the database, there are over 10TB high quality data generated for the research community.

Source: https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Multimodal_Spontaneous_Emotion_CVPR_2016_paper.pdf  for the database details.

# References
Frédéric Bousefsaf et al., Estimation of blood pressure waveform from facial video using a deep U-shaped
network and the wavelet representation of imaging photoplethysmographic signals, Biomedical Signal Processing and Control, 2022.

# Requirements
Different packages must be installed to properly run the codes :

- `pip install opencv-python`
- `pip install numpy`
- `pip install plotly`
- `pip install os`
- `pip install pandas`
- `pip install importlib`
- `pip install abc`
- `pip install numba`
- `pip install mediapipe`
- `pip install matplotlib`
- `pip install scipy`
- `pip install scikit-learn`
- `pip install cupy`
- `pip install torch`
- `pip install requests`
- `pip install segmentation-models`

work in progress...
