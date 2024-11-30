from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pickle as pk
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import datetime as dt
import wikipediaapi

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

PARENT_DIR = os.path.join(FILE_DIR, os.pardir)

dir_of_interest = os.path.join(PARENT_DIR, "files")

cv_path = os.path.join(dir_of_interest,  "cv.pkl")
rg_path = os.path.join(dir_of_interest,  "rg.pkl")
vectors_path = os.path.join(dir_of_interest,  "vectors.pkl")

rg = pk.load(open(rg_path, 'rb'))


victim = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
            'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
            "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
            'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "ff",
            "suffering", "I", "and"
        }


def preprocess_text(text):
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    tokens = [word for word in tokens if word not in victim]


    return tokens





class ActionGetDiagnosis(Action):
    def name(self) -> Text:
        return "action_get_diagnosis"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get("text")

        preprocessed_symptoms = preprocess_text(user_input)
        preprocessed_symptoms = " ".join(preprocessed_symptoms)


        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=5000, stop_words="english")

        vectors = cv.fit_transform(rg["All_Symptoms"])

        symptom_vector = cv.transform([preprocessed_symptoms])


        similar = cosine_similarity(vectors, symptom_vector)


        index = similar.argmax()


        diagnosis = rg.loc[index, "diagonis"]
        disease = rg.loc[index, "Disease"]


        dispatcher.utter_message(text=f"According to your symptoms, you may have {disease}. || \n"
                                      f"The diagnosis of {disease} is {diagnosis}. || \n Did you find this helpful?")

        return []

class ActionDiseaseInfo(Action):
    def name(self) -> Text:
        return "action_disease_info"




    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        disease = tracker.get_slot("disease")
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page = wiki_wiki.page(disease)

        if page.exists():
            info = page.summary[0:1000]
            dispatcher.utter_message(
                text=f"Here is some information about {disease} => \n{info} || Is this helpful ?"
            )

        else:

            dispatcher.utter_message(
                text="I'm sorry, I couldn't find information about that disease. Please try a different "
                     "disease or use the disease with a correct spelling"
            )

        return []

class ActionGettime(Action):


    def name(self) -> Text:
        return "action_give_time"

    def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:


        dispatcher.utter_message(text=f"The current time is - {dt.datetime.now()}")

        return []
