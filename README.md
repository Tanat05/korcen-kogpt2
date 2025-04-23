<div align="center">
  <h1>Korcen-kogpt2</h1>
  <h2>This failure is the seed of innovation.</h2>
</div>

<p align="center">
  <img src="https://user-images.githubusercontent.com/85154556/171998341-9a7439c8-122f-4a9f-beb6-0e0b3aad05ed.png" alt="131_20220604170616">
</p>

"Intelligent Filtering: Detecting Nuance and Context with Machine Learning."

Moving beyond keyword matching, this project introduces a machine learning-powered profanity filter. By analyzing context and linguistic patterns, it aims to identify and filter out offensive language more accurately and intelligently, even when subtle variations or creative spellings are used.

[Korcen](https://github.com/KR-korcen/korcen): original before innovation.

[Korcen-13M-EXAONE](https://github.com/Tanat05/Korcen-13M-EXAONE): This failure, though another, is a better one.

# Model Overview
```
total samples: 2,000,000
Training samples: 1,800,000
Validation samples: 200,000
```

Tokenizer: SKT-AI/KoGPT2


# Verification

<div align="center">

| 모델                                                       | [korean-malicious-comments-dataset](https://github.com/ZIZUN/korean-malicious-comments-dataset) | [Curse-detection-data](https://github.com/2runo/Curse-detection-data) | [kmhas_korean_hate_speech](https://huggingface.co/datasets/jeanlee/kmhas_korean_hate_speech) | [Korean Extremist Website Womad Hate Speech Data](https://www.kaggle.com/datasets/captainnemo9292/korean-extremist-website-womad-hate-speech-data/data) | [LGBT-targeted HateSpeech Comments Dataset (Korean)](https://www.kaggle.com/datasets/junbumlee/lgbt-hatespeech-comments-at-naver-news-korean) |
| :--------------------------------------------------------- | ----------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------: |
| [korcen](https://github.com/KR-korcen/korcen)             |                                                                                             0.7121 |                                                                                              0.8415 |                                                                                                   0.6800 |                                                                                                                                   0.6305 |                                                                                                                               0.4479 |
| TF [VDCNN_KOGPT2](https://github.com/KR-korcen/korcen-ml/tree/main/model) (23.06.15) |                                                                                             0.7545 |                                                                                              0.7824 |                                                                                                          |                                                                                                                                   0.7055 |                                                                                                                               0.6875 |
</div>

## Example

```python
# py: 3.10, tf: 2.10
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 1000

model_path = 'vdcnn_model.h5'
tokenizer_path = "tokenizer.pickle"

model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    return text

def predict_text(text):
    sentence = preprocess_text(text)
    encoded_sentence = tokenizer.encode_plus(
        sentence,
        max_length=maxlen,
        padding="max_length",
        truncation=True
    )['input_ids']

    sentence_seq = pad_sequences([encoded_sentence], maxlen=maxlen, truncating="post")
    prediction = model.predict(sentence_seq)[0][0]
    return prediction

while True:
    text = input("Enter the sentence you want to test: ")
    result = predict_text(text)
    if result >= 0.5:
        print("This sentence contains abusive language.")
    else:
        print("It's a normal sentence.")
```

- [SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2)
- [[NDC] 딥러닝으로 욕설 탐지하기](https://youtu.be/K4nU7yXy7R8)
- [머신러닝 부적절 텍스트 분류:실전편](https://medium.com/watcha/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%B6%80%EC%A0%81%EC%A0%88-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EB%B6%84%EB%A5%98-%EC%8B%A4%EC%A0%84%ED%8E%B8-57587ecfae78)
