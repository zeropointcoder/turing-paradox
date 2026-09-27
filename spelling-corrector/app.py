import streamlit as st
from collections import Counter
import re
import itertools


class UKCorpus:
    def __init__(self):
        self.text = """
        colour organise behaviour centre theatre defence offence licence practise programme travelling cancelled modelling jewellery labour neighbour favourite humour flavour honour metre litre catalogue aluminium analyse apologise aeroplane biscuit cheque cosy grey marvellous plough travelled tyre yoghurt
        """

    def words(self):
        return re.findall(r"[a-z]+", self.text.lower())


class SpellModel:
    def __init__(self, corpus_words):
        self.frequencies = Counter(corpus_words)
        self.vocab = set(self.frequencies.keys())
        self.total = sum(self.frequencies.values())

    def probability(self, word):
        # Unigram probability
        return self.frequencies[word] / self.total if word in self.frequencies else 0.0


class EditGenerator:
    def __init__(self, vocab):
        self.vocab = vocab
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
        deletes = [L+R[1:] for L,R in splits if R]
        transposes = [L+R[1]+R[0]+R[2:] for L,R in splits if len(R)>1]
        replaces = [L+c+R[1:] for L,R in splits if R for c in self.alphabet]
        inserts = [L+c+R for L,R in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def known(self, words):
        return set(w for w in words if w in self.vocab)


class SpellCorrector:
    def __init__(self, model, editor):
        self.model = model
        self.editor = editor

    def correct(self, word):
        word = word.lower()
        if word in self.model.vocab:
            return word

        # Candidates: word itself, edit-distance 1, then edit-distance 2
        candidates = (
            self.editor.known([word]) or
            self.editor.known(self.editor.edits1(word)) or
            self.editor.known(self.editor.edits2(word)) or 
            [word]
        )

        return max(candidates, key=self.model.probability)


class StreamlitUI:
    def __init__(self):
        corpus = UKCorpus()
        model = SpellModel(corpus.words())
        editor = EditGenerator(model.vocab)
        self.corrector = SpellCorrector(model, editor)

    def run(self):
        st.set_page_config(page_title="UK Spelling Corrector", layout="centered")
        st.title("UK English Spelling Corrector")
        text = st.text_input("Enter a word")
        if text:
            corrected = self.corrector.correct(text)
            st.markdown(f"**Suggested correction:** `{corrected}`")


if __name__ == "__main__":
    StreamlitUI().run()