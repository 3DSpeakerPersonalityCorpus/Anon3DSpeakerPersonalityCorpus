The SONICOM 3D Speaker Personality Corpus is a dataset designed to support research of spatial audio environments, speech-based personality perception and distance estimation. It contains 360 audio recordings of 120 speakers (60 male, 60 female) reading aloud at three distinct distances (0.5m, 2m, 5m). The corpus can be downloaded at the following link: https: //github.com/S3DSPC/Sonicom3DSpeakerPersonalityCorpus. 

The distribution includes the following materials:
- Audio recordings (720 items);
- k-fold lists (k = 4 lists corresponding to the folds to be
used in a k-fold protocol);
- Python scripts for classification, regression, and feature
extraction (4 items);
- Acoustic features of stimuli (140 items);
- Acoustic features averaged by speaker (1 item);
- Listener personality ratings of stimuli (1 item);
- Listener personality ratings averaged by speaker (6
items);
- Listener distance ratings of stimuli (1 item); • README file (1 item);
- How to cite file (1 item)**Directory Structure****File Naming Convention**- S: Stimulus- nnn: Speaker ID (a unique identifier for each speaker, e.g., 001, 002, ..., 120)- G: Gender (M for male, F for female)- L: Location (G for University of Glasgow, I for Imperial College London)- D: Distance (N for near-field (0.5m), C for conversational (2m) and F for further (5m))- C: Channel (C1 for headphone channel 1, C2 for headphone channel 2)
Example: S047MGN1 (Stimulus, Speaker 047, Male, Glasgow, Near-field, Channel 1)**Data Overview** Note: for a full description of the corpus, please refer to O'Hara et al., 2024 (DOI: 0.xxxx/xxxxx)[unpublished].This dataset consists of 720 audio recordings from 120 native English speakers (60 male, 60 female), recruited equally from the University of Glasgow and Imperial College London. Participants recorded Aesop’s fable "The North Wind and the Sun" (see below) in anechoic booths using a RØDE NT-USB microphone. Each speaker read the story three times, simulating different listener distances: near-field (0.5m), conversational (2m), and further (5m). The recordings were truncated to 10 seconds and normalised for consistency.
Speech stimuli were assessed by a separate group of 10 listeners (5 female, 5 male) on the Big Five personality traits of openness, conscientiousness, extraversion, agreeableness, and neuroticism (on a 0 to 100 scale) and estimated distance (on a 0 to 10m scale). 

**Contact Information**To get in touch for questions about the corpus please contact alessandro.vinciarelli@glasgow.ac.uk

**Speech materials**

The North Wind and the Sun [1]**

The North Wind and the Sun were disputing which was the stronger, when a
traveller came along wrapped in a warm cloak. They agreed that the one who
first succeeded in making the traveller take his cloak off should be considered
stronger than the other. Then the North Wind blew as hard as he could, but the
more he blew the more closely did the traveller fold his cloak around him; and
at last the North Wind gave up the attempt. Then the Sun shone out warmly,
and immediately the traveller took his cloak off. And so the North Wind was
obliged to confess that the Sun was the stronger of the two. 

[1] International Phonetic Association, Handbook of the International Phonetic Association: A guide to the use of the International Phonetic Alphabet, Cambridge: Cambridge University Press, 1999.