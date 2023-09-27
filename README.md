# Assessment of Non-Native Pronunciation

## This project was the final Group Project submission for the Applied Machine Learning class at UTD.


## Table of Contents
1. [Introduction](#introduction)
2. [Why This Topic?](#why-this-topic)
3. [Why Should We Care?](#why-should-we-care)
4. [Evolution - Need for Standardization](#evolution-need-for-standardization)
5. [Business Side Story](#business-side-story)
6. [Evolution of CAPT through Time](#evolution-of-capt-through-time)
7. [Our Approach for the Problem](#our-approach-for-the-problem)
8. [Key Terms to Know](#key-terms-to-know)
9. [Motivation](#motivation)
10. [ASR Evolution](#asr-evolution)
11. [Introduction to ASR](#introduction-to-asr)
12. [History of Automatic Speech Recognition](#history-of-automatic-speech-recognition)
13. [ASR Architecture](#asr-architecture)
14. [ASR Evolution: GMM-HMM to End-to-End Deep Models](#asr-evolution-gmm-hmm-to-end-to-end-deep-models)
15. [Automatic Pronunciation Assessment (APA)](#automatic-pronunciation-assessment-apa)
16. [APA: Goodness of Pronunciation](#apa-goodness-of-pronunciation)
17. [Advanced GOPT with Multi-view](#advanced-gopt-with-multi-view)
18. [Wav2Vec 2.0](#wav2vec-20)
19. [Wav2Vec Architecture](#wav2vec-architecture)
20. [Wav2Vec Training](#wav2vec-training)
21. [Wav2Vec Results](#wav2vec-results)
22. [Wav2Vec Demo](#wav2vec-demo)
23. [Conclusion](#conclusion)

---

## Introduction<a name="introduction"></a>
Welcome to the "Assessment of Non-Native Pronunciation" project! In this README, we will provide an overview of our advanced deep learning project and its significance. This project aims to assess non-native pronunciation using state-of-the-art techniques in Automatic Speech Recognition (ASR) and Goodness of Pronunciation (GoP) scoring.

---

## Why This Topic?<a name="why-this-topic"></a>
1. **Wide Area of Implementation:** Our project addresses a real-world problem with a broad range of applications, particularly in language education and assessment. It has the potential to automate tasks like evaluating pronunciation in exams such as TOEFL and IELTS.

2. **Extremely Relatable for International Students:** As pronunciation is a crucial aspect of language learning, our project is highly relevant and beneficial for international students striving to improve their language skills.

3. **Something Additional for Class - Wave to Vector:** We introduce innovative techniques like "Wave to Vector" in pronunciation assessment, contributing to the advancement of language learning.

---

## Why Should We Care?<a name="why-should-we-care"></a>
Let's take a look at some statistics highlighting the importance of pronunciation assessment:

- Approximately 1,075,496 international students attended American universities in the 2019–2020 academic year, making up 4.6% of all enrolled students in the country.

- The combined $39 billion in economic contributions made by international students during the 2018–19 academic year supported over 400,000 jobs in the United States.

- International students gave public universities $9 billion in 2015, constituting 28% of their entire revenue.

- STEM programs are attended by about half of the foreign students studying in the United States. And TOEFL/IELTS is required for each and every one of them!

*(Source: [Prosperity for America](https://www.prosperityforamerica.org/international-students-in-the-us/))*

These statistics underscore the significance of assessing non-native pronunciation, especially for international students who make substantial contributions to educational institutions and the economy.

---

## Evolution - Need for Standardization<a name="evolution-need-for-standardization"></a>
Early methods of pronunciation assessment relied heavily on subjective judgments from language teachers, making assessments unreliable and subject to human bias. However, with the growing interest in standardized testing in the 1950s and 1960s, tests like the Michigan English Test and TOEFL were developed to address these issues.

---

## Business Side Story<a name="business-side-story"></a>
Effective pronunciation assessment has significant implications for various business aspects:

- **Language Training and Development:** Ensuring employees' effective communication within an organization.

- **Customer Service and Sales:** Beneficial for businesses dealing with diverse linguistic backgrounds.

- **International Presentations and Public Speaking:** Crucial for engaging and influencing global audiences.

- **Cross-Cultural Communication:** Pronunciation plays a significant role in effective cross-cultural interactions.

- **Professional Image and Confidence:** Computer Assisted Pronunciation Training (CAPT) can help establish a professional image in the business world.

---

## Evolution of CAPT through Time<a name="evolution-of-capt-through-time"></a>
Computer Assisted Pronunciation Training (CAPT) has evolved significantly over the years:

- 1960s-1970s: Early research in phonetics and phonology laid the foundation for CAPT.

- 1980s-1990s: Advancements in speech recognition technology enabled the development of more accurate CAPT systems.

- 1990s-early 2000s: Multimedia and computer-based training facilitated the development of interactive CAPT software.

- 2000s-2010s: Internet and mobile technologies revolutionized CAPT, making it accessible through online platforms and mobile apps.

- 2010s-present: Data-driven approaches, machine learning, deep learning, and integration with other technologies like VR and AR have enhanced CAPT's capabilities.

---

## Our Approach for the Problem<a name="our-approach-for-the-problem"></a>
Our project involves the following key components:

1. **Evolution of Computer Assisted Pronunciation Training (CAPT):** We discuss the historical development of CAPT.

2. **Automatic Speech Recognition (ASR):** We explore the evolution, architecture, and advancements in ASR.

3. **Automatic Pronunciation Assessment (APA):** We delve into the concept of GoP (Goodness of Pronunciation) and its role in pronunciation assessment.

4. **Goodness of Pronunciation Transformer (GOPT):** We present the architecture of GOPT with multi-view features.

5. **Wave to Vector:** An introduction to the wave to vector technique widely used in pronunciation evaluation and mispronunciation detection.

6. **Quick Demo:** A brief overview of our project's capabilities.

---

## Key Terms to Know<a name="key-terms-to-know"></a>
To better understand our project, here are some key terms:

1. **Phoneme:** The smallest units of sound that make up words in spoken language.

2. **Utterance:** A spoken word or group of spoken words, which can be a complete sentence, phrase, or single word.

3. **Prosody:** The rhythmic and melodic aspects of speech that convey emotion.

4. **Grapheme:** The smallest unit of a writing system, such as a letter or symbol, which corresponds to a phoneme in the spoken language.

5. **ASR (Automatic Speech Recognition):** The technology that converts spoken language into written text.

6. **CAPT (Computer Assisted Pronunciation Training):** Software or systems designed to help language learners improve their pronunciation.

7. **GoP (Goodness of Pronunciation):** A measure used to assess the quality of pronunciation in CAPT systems.

8. **GOPT (Goodness of Pronunciation Transformer):** A deep learning model designed for GoP scoring.

9. **Wav2Vec (Wave2Vector):** A deep learning model used for automatic speech recognition.

---

## Motivation<a name="motivation"></a>
The motivation behind our project is to bridge the gap between traditional pronunciation assessment methods and modern technology. By leveraging advanced deep learning techniques, we aim to provide a more objective, consistent, and automated approach to assessing non-native pronunciation. This can greatly benefit language learners, educators, and organizations seeking to improve communication skills.

---

## ASR Evolution<a name="asr-evolution"></a>
Automatic Speech Recognition (ASR) technology has come a long way. Let's take a journey through its evolution.

---

## Introduction to ASR<a name="introduction-to-asr"></a>
ASR, or Automatic Speech Recognition, is the technology that converts spoken language into written text. It's widely used in various applications, including transcription services, voice assistants, and, in our case, pronunciation assessment.

---

## History of Automatic Speech Recognition<a name="history-of-automatic-speech-recognition"></a>
ASR has a rich history:

- **1950s-1960s:** Early research in speech recognition focused on isolated word recognition using techniques like dynamic time warping.

- **1970s-1980s:** The introduction of Hidden Markov Models (HMMs) revolutionized ASR and enabled more accurate continuous speech recognition.

- **1990s-2000s:** The use of Gaussian Mixture Models (GMMs) in conjunction with HMMs improved ASR's performance. This era saw significant commercialization.

- **2010s-present:** Deep learning, particularly deep neural networks (DNNs), recurrent neural networks (RNNs), and convolutional neural networks (CNNs), has taken ASR to new heights.

---

## ASR Architecture<a name="asr-architecture"></a>
ASR systems consist of several key components:

1. **Acoustic Model:** This model captures the relationship between acoustic features (e.g., audio signals) and phonemes.

2. **Language Model:** The language model incorporates linguistic knowledge to predict word sequences.

3. **Decoding:** The decoding process combines information from the acoustic and language models to generate the most likely word sequence.

4. **Training Data:** ASR systems require extensive training data, typically consisting of audio recordings and corresponding transcriptions.

---

## ASR Evolution: GMM-HMM to End-to-End Deep Models<a name="asr-evolution-gmm-hmm-to-end-to-end-deep-models"></a>
The transition from GMM-HMM-based ASR to end-to-end deep models marked a significant shift in the field:

- **GMM-HMM (1990s-2010s):** GMMs modeled acoustic features, while HMMs represented temporal dependencies. This combination worked well but had limitations in modeling complex sequences.

- **Deep Learning (2010s-present):** Deep neural networks, including CNNs and RNNs, have shown remarkable performance improvements. End-to-end models like Listen, Attend, and Spell (LAS) and Connectionist Temporal Classification (CTC) have become popular.

---

## Automatic Pronunciation Assessment (APA)<a name="automatic-pronunciation-assessment-apa"></a>
Automatic Pronunciation Assessment (APA) is a critical part of Computer Assisted Pronunciation Training (CAPT). It involves evaluating the pronunciation quality of a spoken utterance.

---

## APA: Goodness of Pronunciation<a name="apa-goodness-of-pronunciation"></a>
Goodness of Pronunciation (GoP) is a measure used in APA to assess the quality of pronunciation. GoP scoring is based on factors like phoneme accuracy, prosody, fluency, and rhythm.

---

## Advanced GOPT with Multi-view<a name="advanced-gopt-with-multi-view"></a>
Our project introduces an advanced Goodness of Pronunciation Transformer (GOPT) model with multi-view features. This model leverages deep learning to provide more accurate GoP scores.

---

## Wav2Vec 2.0<a name="wav2vec-20"></a>
Wav2Vec 2.0 is a deep learning model designed for automatic speech recognition. It's widely used in ASR tasks, including pronunciation assessment.

---

## Wav2Vec Architecture<a name="wav2vec-architecture"></a>
Wav2Vec 2.0 is built upon the CTC (Connectionist Temporal Classification) framework and uses convolutional neural networks (CNNs) to learn acoustic representations from audio signals.

---

## Wav2Vec Training<a name="wav2vec-training"></a>
Training Wav2Vec 2.0 involves pretraining on a large corpus of multilingual and multitask supervised data. Fine-tuning is then performed on specific ASR tasks to adapt the model for pronunciation assessment.

---

## Wav2Vec Results<a name="wav2vec-results"></a>
Wav2Vec 2.0 has achieved impressive results in various ASR benchmarks, making it a powerful tool for automatic pronunciation assessment.

---

## Wav2Vec Demo<a name="wav2vec-demo"></a>
Want to see Wav2Vec in action? Stay tuned for our project demo, where we'll showcase the model's capabilities in assessing non-native pronunciation.

---

## Conclusion<a name="conclusion"></a>
In conclusion, our "Assessment of Non-Native Pronunciation" project combines the power of Automatic Speech Recognition (ASR) with advanced Goodness of Pronunciation (GoP) scoring to provide an objective and automated solution for evaluating non-native pronunciation. This has immense potential in language education, standardized testing, and various industries where effective communication is essential. Stay tuned for more updates and our project demo!
