import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class EmailDataset:
    def __init__(self):
        self.emails = [
            # --- request ---
            "Could you please send me the updated report by tomorrow?",
            "I would appreciate it if you could share the latest figures.",
            "Please provide the final version of the document.",
            "Can you send over the details when you have a moment?",
            "I am requesting access to the shared folder.",
            "Please let me have the revised draft today.",
            "Could you forward the information discussed earlier?",
            "I would be grateful if you could send the invoice.",
            "Please supply the missing data points.",
            "Can you provide clarification on the figures?",

            # --- review ---
            "Please review the attached document and share your feedback.",
            "I have attached the proposal for your review.",
            "Could you look over the report and let me know your thoughts?",
            "Please check the document for accuracy.",
            "I would appreciate your feedback on the attached file.",
            "Kindly review the draft and suggest any changes.",
            "Please examine the document before final submission.",
            "Can you review the presentation slides?",
            "Let me know if you spot any issues in the document.",
            "Please validate the figures in the attached report.",

            # --- meeting ---
            "Can we schedule a meeting for next week?",
            "Please let me know a suitable time for a meeting.",
            "Shall we arrange a call to discuss this further?",
            "I would like to set up a meeting to go through the details.",
            "Are you available for a meeting on Thursday?",
            "Let us arrange a meeting at your convenience.",
            "Could we book some time to discuss the proposal?",
            "Please confirm your availability for a meeting.",
            "Can we organise a quick call tomorrow?",
            "I suggest a meeting to align on next steps.",

            # --- enquiry ---
            "I would like to enquire about your services.",
            "Could you provide information on your pricing?",
            "I am interested in learning more about your offerings.",
            "Please let me know what options are available.",
            "I would appreciate further details about your service.",
            "Can you explain how your process works?",
            "I am enquiring about your availability.",
            "Could you share more information on this?",
            "Please advise on the next steps.",
            "I would like to know more about your packages.",

            # --- follow_up ---
            "I am following up on my previous email.",
            "Just checking in regarding my last message.",
            "I wanted to follow up to see if there is any update.",
            "Please let me know if you have had a chance to review this.",
            "I am writing to follow up on our earlier discussion.",
            "Could you provide an update on this matter?",
            "Following up to see if you need anything further.",
            "Just a quick follow-up on my earlier request.",
            "I am awaiting your response regarding this.",
            "Please advise if there has been any progress."
        ]

        self.labels = (
            ["request"] * 10 +
            ["review"] * 10 +
            ["meeting"] * 10 +
            ["enquiry"] * 10 +
            ["follow_up"] * 10
        )


class EmailIntentModel:
    def __init__(self):
        self.vectoriser = TfidfVectorizer(stop_words="english")
        self.classifier = LogisticRegression(max_iter=1000)

    def train(self, texts, labels):
        vectors = self.vectoriser.fit_transform(texts)
        self.classifier.fit(vectors, labels)

    def predict(self, text):
        vector = self.vectoriser.transform([text])
        return self.classifier.predict(vector)[0]


class ResponseGenerator:
    def __init__(self):
        self.responses = {
            "request": "Thank you for your email. I will review your request and respond shortly.",
            "review": "Thank you for sharing the document. I will review it and provide feedback soon.",
            "meeting": "Thank you for reaching out. I will confirm a suitable meeting time shortly.",
            "enquiry": "Thank you for your enquiry. I will get back to you with the relevant details.",
            "follow_up": "Thank you for following up. I will respond once I have an update."
        }

    def generate(self, intent):
        return self.responses.get(
            intent,
            "Thank you for your email. I will get back to you shortly."
        )


class SmartMailerApp:
    def __init__(self):
        self.dataset = EmailDataset()
        self.model = EmailIntentModel()
        self.responder = ResponseGenerator()
        self.model.train(self.dataset.emails, self.dataset.labels)

    def run(self):
        st.set_page_config(page_title="Smart Mailer", layout="centered")
        st.title("Smart Mailer: ML Email Responder")

        email_text = st.text_area("Enter your email", "")

        if st.button("Generate response"):
            if not email_text.strip():
                st.warning("Please enter an email.")
                return

            intent = self.model.predict(email_text)
            response = self.responder.generate(intent)

            st.subheader("Detected intent")
            st.write(intent)

            st.subheader("Suggested response")
            st.write(response)


if __name__ == "__main__":
    SmartMailerApp().run()