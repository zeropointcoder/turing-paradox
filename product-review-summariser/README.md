# Product Review Summariser

An ML-based product review summariser using TF-IDF and cosine similarity to extract top sentences.


## Overview
- Clean and preprocess text: `lowercase`, remove `punctuation`, `split` into sentences.

- Vectorise sentences with `TF-IDF` to capture word `importance` statistically.

- `Score` sentences by computing their `similarity` to the overall review corpus.

- `Rank` sentences and return the top `N` as the summary.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

<!--
Input sample review:

I recently bought the Acme headphones and I am absolutely thrilled with them. The sound quality is crisp and clear, with excellent bass. The battery life is impressive, lasting more than 20 hours on a single charge. However, the ear cushions are a bit tight and may be uncomfortable for long listening sessions. Overall, these headphones are fantastic for daily use and commuting.

The build quality is solid and feels premium. I also appreciate the sleek design, which looks very modern. Noise cancellation works reasonably well but struggles in very noisy environments. Customer service was helpful when I had a question about pairing. Definitely worth the price for anyone looking for quality sound.

I love the wireless connectivity feature; pairing with my devices is effortless. The headphones are lightweight and easy to carry around. On the downside, the touch controls can be overly sensitive and sometimes skip tracks accidentally. Still, I would recommend these to anyone who enjoys music on the go.
-->