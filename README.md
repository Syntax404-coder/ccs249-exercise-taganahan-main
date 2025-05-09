# ccs249-exercise-taganahan-main

This project implements a **Hidden Markov Model (HMM)** for **Part-of-Speech (POS) tagging**. It uses a small tagged dataset to train the model, which learns transition and emission probabilities. The model then uses the **Viterbi algorithm** to predict the most likely tag sequence for new sentences. The main steps involve:

1. Preprocessing sentences with start and end tokens.
2. Calculating transition and emission probabilities.
3. Training the HMM on the data.
4. Predicting tags for test sentences.

Example:
For "The cat meows", the model predicts tags like `['DET', 'NOUN', 'VERB']`.

This HMM implementation can be extended for larger datasets and more complex tasks.
