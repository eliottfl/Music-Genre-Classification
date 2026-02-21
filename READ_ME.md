# Automatic Music Genre Classification (TIPE Project)

**Candidate:** Eliott FLAMENT  
**Program:** Highly Selective Two-Year Program preparing for the national entrance exams to top French Engineering Schools.  
**Grade received:** 15.8 / 20  
**Problem statement:** Can we train a model capable of automatically and reliably recognizing the musical genre of a track?

This project studies the classification of music tracks using several supervised learning algorithms.

## 1. Dataset and Extraction

The project uses the **FMA Small** dataset:
* 8,000 30-second tracks.
* 8 main genres (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock) evenly distributed.

### Multiprocessing Optimization
Extracting audio features is computationally intensive. To speed up the process, I implemented **multiprocessing** in `extraction.py`:
I used the `multiprocessing.Pool` library to parallelize the analysis of the mp3 files.
* Extraction of 33 features per track.

## 2. Data Preprocessing

To optimize learning, the data is cleaned using two scripts:
* **Correlation (`correlation.py`):** Removal of variables correlated at more than 95%.
* **Variance (`variance.py`):** Removal of variables with a variance lower than 0.005.

## 3. Model Comparison

Four models were trained and optimized to compare their accuracy:

| Model | Accuracy |
| :--- | :--- |
| **k-NN** | **39%** |
| **Decision Tree** | **35%** |
| **Random Forest** | **47%** |
| **XGBoost** | **48%** |

## 4. Conclusion and Analysis

Variable importance analysis shows that the spectral_centroid, percussive_rms, and mfcc 1 to 3 features are consistently decisive in the classification. 
All models struggle to predict when a track is experimental or pop.

This project allowed me to dive into Machine Learning and Data Science, a field I’ve been passionate about for a few years now. It was a great opportunity to apply what I learned in my machine learning lectures with k-NN and decision trees, then moving on to more advanced techniques like Random Forest and XGBoost. I’m really happy with the project, the results, and everything I’ve learned. I’m excited to keep pushing further.

## 5. Installation and Usage

1. **Dependencies:** `pip install pandas numpy librosa matplotlib seaborn scikit-learn xgboost tqdm`
2. **Pipeline:**
   - Extract data: `python extraction.py`
   - Filter: `python correlation.py` then `python variance.py`
   - Train: `python model_xgboost.py`