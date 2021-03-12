import pandas as pd 
import pickle
import streamlit as st
import re
from nltk.corpus import stopwords

# Display on webpage
st.title('Movie review')
st.markdown("This webpage uses the IMDB movie review dataset")
st.markdown("Using sentiment analysis, reviews will be classified under a specific category")
st.sidebar.title("Steps: ")
st.sidebar.markdown("1. Type your opinion")
st.sidebar.markdown("2. Press enter")
st.sidebar.markdown("3. Wait for the respective emoji to appear")

# Load previously created models
loaded_model = pickle.load(open("MNBmodel.pkl","rb"))
cv = pickle.load(open("countvectorizer.pkl","rb"))

# Predict output for new review
def new_review(new_review):
  new_review = new_review
  new_review = re.sub('[^\w\s]', ' ', new_review)
  new_review = new_review.lower()
  new_review = new_review.split()
  stop_words = stopwords.words('english')
  new_review = [word for word in new_review if not word in set(stop_words)]
  new_review = ' '.join(new_review)
  new_corpus = [new_review]
  new_x_test = cv.transform(new_corpus).toarray()
  new_y_pred = loaded_model.predict(new_x_test)
  return new_y_pred

# Get user input
input_review = st.text_input('Enter new review:')
new_review = new_review(input_review)
if new_review[0]==1:
   st.title(":smile:")
else :
   st.title(":worried:")
