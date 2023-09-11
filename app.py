from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def words_tokenize(text):
  text = text.split()
  return text

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lem_word(text):
  # 1. Init Lemmatizer
  lemmatizer = WordNetLemmatizer()
  # 3. Lemmatize with the appropriate POS tag
  return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text]


app = Flask(__name__)

model = pickle.load(open("pa/reg_logit_model.pkl", "rb"))
multilabel_binarizer = pickle.load(open("pa/multilabel_binarizer.pkl", "rb"))
vectorizer = pickle.load(open("pa/vectorizer.pkl", "rb"))


@app.route('/predict',methods=['POST'])
def predict():

    data = request.get_json(force=True)
    print('data=', data)

    data_clean= words_tokenize(data['question'])
    data_clean= lem_word(data_clean)
    #data_clean= " ".join(data_clean)
    print('data_clean= ', data_clean)

    X_input = vectorizer.transform([data_clean]).toarray()
    print ('X_input=', X_input.sum())



    y_pred = model.predict_proba(X_input)
    print('y_pred=', y_pred)

    threshold=0.05
    y_pred = (y_pred > threshold) * 1


    y_pred_inversed = multilabel_binarizer.inverse_transform(y_pred)
    print('y_pred_inversed=', y_pred_inversed)

    response = {"predicted_tags": y_pred_inversed[0]}


    return response




if __name__ == '__main__':
 app.run(port=5000, debug=True)
 print('start api')