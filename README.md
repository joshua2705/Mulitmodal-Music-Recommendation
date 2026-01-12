# Content-Based Music Recommendation using Autoencoder

This project implements a **content-based** music recommendation system that operates purely on audio content, without relying on user interaction history or popularity signals. Short song previews are converted to mel-spectrogram images, encoded into a latent space using a convolutional autoencoder, and recommendations are produced via cosine similarity in that latent space.

## Team

- Olga NG  
- Joshua Alexander EBENEZER  
- George EMMANUEL

## 1. Introduction / Motivation

Content-based recommendation systems analyze the actual audio properties of songs rather than user behavior, enabling recommendations based on musical similarity regardless of popularity or historical play counts. This project focuses on deep learning–based extraction of acoustic features so that similar tracks can be identified purely from their audio signals, helping users discover musically similar but potentially underrated or new songs.

**Key goals:**
- Learn meaningful audio embeddings from mel-spectrograms  
- Retrieve similar songs using cosine similarity in the learned latent space  
- Evaluate whether recommendations feel musically relevant to listeners via user study

## 2. Constraints and Practical Restrictions

**Dataset size:**
- Initially targeted 200 random pop artists from Spotify (≈2000 songs)
- Final working dataset: 1,619 usable tracks after filtering

**Filtering criteria:**
- Removed non-English songs
- Removed songs released before 2000
- Removed tracks without available audio previews

**Compute limitations:**
- Training restricted to 600 tracks
- Inputs resized to 128×256 resolution

## 3. Methodology

### 3.1 Overview

`![Recommendation Pipeline](images/figure1_pipeline.png)`

### 3.2 Data Collection

1. **Spotify API**: Scraped metadata for 200 pop artists (top 10 songs each)
2. **Deezer API**: Retrieved audio preview MP3s (filtered tracks without previews)
3. **Storage**: Feather file with track ID, name, artist, preview URL

### 3.3 Audio Preprocessing

MP3 decode → Mono channel → Fixed duration → Mel-spectrogram → Log compression


**Mel-spectrogram formula:**
```
Melspectrogram = log(1 + α × spectrogram)
```
where α is a scaling factor.

`![Audio to Mel-spectrogram](images/figure2_melspec.png)`

### 3.4 Autoencoder Learning

**Architecture:**

 `![Autoencoder Architecture](images/figure3_autoencoder.png)`

### 3.5 Latent Embedding Extraction

`![Latent Space Visualization](images/figure4_latent_space.png)`

### 3.6 Similarity Algorithm

**Cosine similarity:**
```
cosine_similarity = A · B / (||A|| × ||B||)
```


**Recommendation steps:**
1. Extract embedding zₖ for query song k
2. Compute sim(k,j) for all other songs j
3. Rank by descending similarity
4. Return top K=20 recommendations

## 4. Evaluation

### 4.1 User Study (17 participants)

Each participant:
1. Selected a liked seed song
2. Received recommendations via cosine similarity
3. Listened and marked "loved" tracks
4. **Love rate = Loved/Listened**

### 4.2 Results

| Person | Seed Song | Artist | Recs | Listened | Love Rate |
|--------|-----------|--------|------|----------|-----------|
| 1 | Pump It | Black Eyed Peas | 11 | 7 | 0.636 |
| 2 | Maps | Maroon 5 | 17 | 12 | 0.706 |
| 3 | Apple | Charli XCX | 10 | 7 | 0.700 |
| 4 | Style | Taylor Swift | 10 | 8 | 0.800 |
| 5 | Brooklyn Baby | Lana Del Rey | 14 | 7 | 0.500 |
| 6 | Feel Good Inc. | Gorillaz | 9 | 7 | 0.778 |
| 7 | Levitating | Dua Lipa | 13 | 12 | 0.923 |
| 8 | Wake Me Up... | Green Day | 16 | 7 | 0.438 |
| 9 | Hung Up | Madonna | 10 | 5 | 0.500 |
| 10 | Enemy | Imagine Dragons | 18 | 12 | 0.667 |
| ... | ... | ... | ... | ... | ... |
| **Avg** | | | | | **0.684** |

## 5. Conclusions

✅ **68.4% average acceptance rate** proves audio embeddings capture musical coherence

✅ Recommendations often matched **mood/tempo** of seed songs

⚠️ Some misses due to unrepresentative previews or mel-spectrogram limitations

## Setup Instructions

1. **Extract images** from PDF (Figures 1-5) → Save in `images/` folder
2. **Update image paths** in markdown links above
3. **Copy this content** → Save as `README.md`

## References

1. Aucouturier & Pachet (2002). Music similarity measures
2. Van den Oord et al. (2013). Deep content-based music recommendation
3. Halloum (2017). Content-based music recommendation using autoencoders
4. Goodfellow et al. (2016). *Deep Learning* (MIT Press)
5. McFee et al. (2015). librosa: Audio analysis in Python

**Aappendix:** `![Autoencoder Code](images/figure5_code.png)`

