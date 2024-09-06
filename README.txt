The Sonicom 3D Speaker Personality Corpus is a dataset designed to support research of spatial audio environments, speech-based personality perception and distance estimation. It contains 360 audio recordings of 120 speakers (60 male, 60 female) reading aloud at three distinct distances (0.5m, 2m, 5m). **Directory Structure**- Audio files- Metadata 	- Folder of spatial features of individual audio stimuli tables	- Table of spatial features for individual speakers (channel 1)
	- Table of spatial features for individual speakers (channel 2) 	- Table of external personality ratings of speakers
	- Table of external personality ratings of stimuli	- Table of perceived distance ratings	- Folder of K folds lists - Python scripts	- Feature extraction	- DNN regression	- DNN classification	- ML classification- READ ME - How to cite **File Naming Convention**- S: Stimulus- nnn: Speaker ID (a unique identifier for each speaker, e.g., 001, 002, ..., 120)- G: Gender (e.g., M for male, F for female)- L: Location (e.g., G for Glasgow, I for Imperial)- D: Distance (e.g., N for near-field (0.5m), C for conversational (2m) and F for further (5m))Example: S047MGN1 (Speaker 047, Male, Glasgow, Near-field, Channel 1)**Data Overview** Note: for a full description of the corpus, please refer to xxx et al., 2024 (DOI: 0.xxxx/xxxxx).This dataset consists of 360 audio recordings from 120 native English speakers (60 male, 60 female), recruited equally from Imperial College London and the University of Glasgow. Participants recorded Aesop’s fable "The North Wind and the Sun" (see below) in pseudo-anechoic booths using a RØDE NT-USB microphone. Each speaker read the story three times, simulating different listener distances: near-field (0.5m), conversational (2m), and further (5m). The recordings were truncated to 10 seconds and normalised for consistency.Speech stimuli were assessed by a separate group of 15 listeners (9 female, 6 male) for Big Five (BF) personality traits (on a 0 to 100 scale) and perceived distance (on a 0 to 10m scale). Each stimulus received ratings from 10 listeners, and the final assessments are the average of these ratings. 

**Contact Information**To get in touch for questions about the corpus please contact xxxx@gla.ac.uk


**The North Wind and the Sun [1]**

The North Wind and the Sun were disputing which was the stronger, when a
traveller came along wrapped in a warm cloak. They agreed that the one who
first succeeded in making the traveller take his cloak off should be considered
stronger than the other. Then the North Wind blew as hard as he could, but the
more he blew the more closely did the traveller fold his cloak around him; and
at last the North Wind gave up the attempt. Then the Sun shone out warmly,
and immediately the traveller took his cloak off. And so the North Wind was
obliged to confess that the Sun was the stronger of the two. 

[1] International Phonetic Association, Handbook of the International Phonetic Association: A guide to the use of the International Phonetic Alphabet, Cambridge: Cambridge University Press, 1999.