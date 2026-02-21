import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

tracks = pd.read_csv("data\\fma_metadata\\tracks.csv", index_col=0, header=[0, 1])

#convert to dictionary to reduce memory usage in multiprocessing
genres = tracks[("track", "genre_top")].to_dict()

def extract_features(file_path):
    """Exctracts features from an audio file and returns a dictionnary of their values"""
    try:
        y, sr = librosa.load(file_path, sr=None)

        S = np.abs(librosa.stft(y))
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        #mean is required to transform the (1,T) matrix into a representative value of the track
        features = {
            "rms": np.mean(librosa.feature.rms(S=S)),
            "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)),
            "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)),
            "spectral_flatness": np.mean(librosa.feature.spectral_flatness(S=S)),
            "harmonic_rms": np.mean(librosa.feature.rms(y=y_harmonic)),
            "percussive_rms": np.mean(librosa.feature.rms(y=y_percussive)),
            "tempo": tempo[0]
        }

        S_db = librosa.power_to_db(S**2)
        mfccs = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=13)
        mfccs_means = np.mean(mfccs, axis=1)
        for i in range(len(mfccs_means)):
            mfcc = mfccs_means[i]
            features["mfcc_" + str(i + 1)] = mfcc

        chromas = librosa.feature.chroma_stft(S=S**2, sr=sr)
        chromas_means = np.mean(chromas, axis=1)
        for i in range(len(chromas_means)):
            chroma = chromas_means[i]
            features["chroma_" + str(i + 1)] = chroma

        return features

    except Exception as e:
        print(f"Features extraction error for {file_path}: {e}")
        return None

def get_labeled_features(file_path):
    """Call extract_features() on the file and add the track_id and the genre of the audio file and returns a dictionnary"""
    path = Path(file_path)
    try:
        #extraction of track_id (name of the file)
        track_id = int(path.stem)
        
        #extraction of genre
        genre = genres.get(track_id)
        if genre is None or pd.isna(genre):
            return None

        features = extract_features(file_path)
        if features:
            features["track_id"] = track_id
            features["genre"] = genre
            return features
    
    except Exception as e:
        print(f"Error for {file_path} : {e}")
        return None
        
    return None

def process_audio(audio_dir):
    """Explore all the files of a folder and call get_labeled_features() on .mp3 files and returns the dataframe"""
    all_files = list(Path(audio_dir).rglob("*.mp3"))

    #multiprocessing
    with Pool(processes=max(1, cpu_count()-4)) as pool:
        
        #tqmd display current advancement of the extraction process
        results = list(tqdm(
            pool.imap(get_labeled_features, all_files), 
            total=len(all_files), 
            desc="Extraction",
            unit="files"
        ))

    #remove the issues which caused a None and create the dataframe
    dataframe = [res for res in results if res is not None]
    return pd.DataFrame(dataframe) 

if __name__ == "__main__":
    audio_dir = "data\\fma_small"
    dataframe = process_audio(audio_dir)
    dataframe.to_csv("data\\features_test.csv", index=False)