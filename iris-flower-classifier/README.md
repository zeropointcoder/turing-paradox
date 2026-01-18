# Iris Flower Classifier

An interactive application that classifies iris flowers into species using measured flower attributes.


## Overview
- Uses the built-in Iris dataset loaded locally from scikit-learn

- Each sample contains four features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width

- Data is standardised using:
  - `z = (x − μ) / σ`

- A multinomial `logistic regression` model is trained:
  - `P(y = k | x) = softmax(wₖx + bₖ)`

- The model outputs:
  - Predicted class label
  - Class probabilities
  
- User inputs are collected via sliders and classified instantly
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```