
## Introduction

The SONICOM 3D Speaker Personality Corpus is a dataset designed to support research of spatial audio environments, speech-based personality perception and distance estimation. It contains 360 audio recordings of 120 speakers (60 male, 60 female) reading aloud at three distinct distances (0.5m, 2m, 5m). 

The corpus can be downloaded at the following link: https: //github.com/S3DSPC/Sonicom3DSpeakerPersonalityCorpus. 

## Directory Structure

The distribution includes the following materials:
- Audio recordings (720 items, 360 for each channel);
- k-fold lists (k = 4 lists corresponding to the folds to be
used in a k-fold protocol);
- Python scripts for classification, regression, and feature
extraction (6 items);
- Acoustic features of stimuli (720 items, 360 for each channel);
- Acoustic features averaged by sample (2 items, 1 for each channel);
- Listener personality ratings by individual judges (1 item);
- Listener distance ratings by individual judges (1 item);
- Listener personality ratings averaged by sample (5 items, 1 for each trait);
- Listener distance ratings averaged by sample (1 item);
- README file (1 item)



## File Naming Convention

- S: Stimulus
- nnn: Speaker ID (a unique identifier for each speaker, e.g., 001, 002, ..., 120)
- G: Gender (M for male, F for female)
- L: Location (G for University of Glasgow, I for Imperial College London)
- D: Distance (N for near-field (0.5m), C for conversational (2m) and F for further (5m))
- C: Channel (C1 for headphone channel 1, C2 for headphone channel 2)

Example: S047MGN1 (Stimulus, Speaker 047, Male, Glasgow, Near-field, Channel 1)

## Data Overview

Note: for a full description of the corpus, please refer to O'Hara et al., 2024 (DOI: 0.xxxx/xxxxx)[unpublished].

This dataset consists of 720 audio recordings from 120 native English speakers (60 male, 60 female), recruited equally from the University of Glasgow and Imperial College London. Participants recorded Aesops fable "The North Wind and the Sun" (see below) in anechoic booths using a RØDE NT-USB microphone. Each speaker read the story three times, simulating different listener distances: near-field (50 cm), conversational (200 cm), and further (500 cm). The recordings were truncated to 10 seconds and normalised for consistency.

Speech stimuli were assessed by a separate group of 10 listeners (5 female, 5 male) on the Big Five personality traits of openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism (on a 0 to 100 scale) and estimated distance (on a 0 to 10m scale). 

## Contact Information
To get in touch for questions about the corpus please contact alessandro.vinciarelli@glasgow.ac.uk

## Speech materials

### The North Wind and the Sun [1]

The North Wind and the Sun were disputing which was the stronger, when a
traveller came along wrapped in a warm cloak. They agreed that the one who
first succeeded in making the traveller take his cloak off should be considered
stronger than the other. Then the North Wind blew as hard as he could, but the
more he blew the more closely did the traveller fold his cloak around him; and
at last the North Wind gave up the attempt. Then the Sun shone out warmly,
and immediately the traveller took his cloak off. And so the North Wind was
obliged to confess that the Sun was the stronger of the two. 

[1] International Phonetic Association, Handbook of the International Phonetic Association: A guide to the use of the International Phonetic Alphabet, Cambridge: Cambridge University Press, 1999.

## How to cite

If you use the SONICOM 3D Speaker Personality Corpus in your research, please cite it as follows:

Bibtex format:
@dataset{ohara2024corpus,Ê
author= {Emily O'Hara and Evangelia Fringe and Nesreen Alshubaily and Lorenzo Picinali
		and Stephen Brewster and Tanya Guha and Alessandro Vinciarelli},
title= {Sonic 3D Speaker Personality Corpus},
year= {2024},
publisher= {unpublished},
DOI= {10.xxxx/xxxxxx},
url= {https://doi.org/10.xxxx/xxxxxx}

ACM Style:
Emily O'Hara, Evangelia Fringi, Nesreen Alshubaily, Lorenzo Picinali, Stephen Brewster, Tanya Guha and Alessandro Vinciarelli. 2024. Sonicom 3D Speaker Personality Corpus. Data set. DOI: https://doi.org/10.xxxxxxxx.