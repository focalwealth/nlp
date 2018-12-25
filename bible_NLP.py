import nltk
#import urlopen
import pandas as pd
#import numpy
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianN beiB
#import os


word_set=[]

def tokenizer_func(data_entry):
    #sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    token=word_tokenize((data_entry).decode('utf-8'))
    #print(token)

    stemmer = PorterStemmer()
    stemmed= [ stemmer.stem(word) for word in token ]
    unique_list= []
    for item in stemmed:
        if item not in unique_list:
            unique_list.append(item)
    # print(len(token))
    # print(len(stemmed))
    #print(len(unique_list))

    stopwords = nltk.corpus.stopwords.words('portuguese')
    #generate stopwords without accents
    stop_symbols = '{}()[].,:;+-*/&!@#$%^_\|%<>=~$1234567890'
    non_sw_list = []
    for words in unique_list:
        if words not in stopwords:
            if any(x in stop_symbols for x in words):
                pass
                # print words, "rejected"
            else:
                non_sw_list.append(words)
                word_set.append(words)
                # print words, "append"


    # print(len(non_sw_list))
    #removing accents here...

    return non_sw_list, stemmed


def cleaner(df):
    data_list = df["Data"].tolist()
    tag_list = df["Tag"].tolist()
    stopwords = nltk.corpus.stopwords.words('portuguese')
    ps = PorterStemmer()
    new = []
    for data in data_list:
        data = re.sub('[^a-zA-Z]', ' ', data)
        data = data.lower()
        data = data.split()
        data = [ps.stem(word) for word in data if not word in stopwords]
        data = " ".join(data)
        new.append(data)

    return new, tag_list


def tf_calculator(basic_words, dumped_words):
    dict = {"word": [], "tf": []}
    for w in basic_words:
        dict["word"].append(w)
        tf = 0
        for token_words in dumped_words:
            if w == token_words:
                tf = tf + 1
        dict["tf"].append(tf)
    return dict
    # if len(dict["tf"])==len(dict("word")):
    #     return pd.DataFrame(dict)

def uniques(listed):
    unique_words = []
    for x in word_set:
        if x not in unique_words:
            unique_words.append(x)
    return unique_words


def run_random_forest_classifier():
    print ("Random forest Classification")
    clfrf = RandomForestClassifier(n_estimators=200, n_jobs=-1, bootstrap= True)
    clfrf.fit(X_train, y_train)
    prediction= clfrf.predict(X_test)
    print ("Model Accuracy for cross validation - ", np.mean(cross_val_score(clfrf, X_test, y_test, cv=10)))
    print ("Accuracy of the model on testing data: {}% ".format(accuracy_score(y_test,prediction)*100))
    print ("F1 score: {}".format(f1_score(y_test,prediction, average= 'micro')))
    print ("Confusion Matrix: ")
    print (confusion_matrix(y_test, prediction))
    print("Below is the prediction")
    print(clfrf.predict(new_test.toarray()))
    return True


def run_nb_classifier():
    print ("Naive Bayes Classification")
    clfnb= GaussianNB()
    clfnb.fit(X_train, y_train)
    prediction = clfnb.predict(X_test)
    print(" new pred")
    print(clfnb.predict(new_test.toarray()))
    print ("Model Accuracy for cross validation - ", np.mean(cross_val_score(clfnb, X_test, y_test, cv=10)))
    print ("Accuracy of the model on testing data: {}% ".format(accuracy_score(y_test, prediction) * 100))
    print ("F1 score: {}".format(f1_score(y_test, prediction, average='micro')))
    print("Confusion Matrix: ")
    print (confusion_matrix(y_test, prediction))
    return True


if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\Sheel\PycharmProjects\\NLP\Data_Set.txt", sep="\t", encoding='latin-1', header=None)
    dsi = {"URL": df[0].tolist(), "Tag": df[1].tolist(), "Data": df[2].tolist()}
    dsi_df = pd.DataFrame(dsi)
    clean_text, tags= cleaner(dsi_df)
    cv = tfidf(sublinear_tf=True, min_df=0.05)
    X = cv.fit_transform(clean_text)
    dtm = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
    X_train=dtm
    y_train=tags
    X_test= dtm.iloc[0:31,:]
    y_test= tags[0:31]
    txt = """Michel Salama HerszageComcast adquire mais de 75% das ações da Sky Operação movimentou US$ 40 bilhões  A Comcast informou que garantiu mais de 75 por cento das ações da Sky, aproximando-se de finalizar a aquisição do grupo britânico de TV paga por 40 bilhões de dólares.  A empresa norte-americana de televisão a cabo Comcast disse anteriormente que espera que a aquisição seja concluída até o final de outubro.  No mês passado, a Comcast emergiu triunfante na longa batalha pela Sky depois de uma disputa com a Twenty-
8:51 PMMichel Salama HerszageFirst Century Fox, de Rupert Murdoch, em um leilão.  A Comcast informou em comunicado nesta quinta-feira que, até 9 de outubro, quando concluir a compra da participação de 39 por cento da Twenty-First Century Fox, já manterá ou terá recebido aceitações em mais de 75 por cento do capital social da Sky.  A empresa disse que um novo anúncio será feito no devido tempo. """
    dft = pd.DataFrame({"Data": [txt], "Tag": ['M']})
    clean_text_test, tags_test = cleaner(dft)
    new_test = cv.transform(clean_text_test)
    # print(new_test)
    run_random_forest_classifier()
    run_nb_classifier()
    print (dtm.shape)
    check=list(set(dsi_df["Tag"]))
    for item in check:
        print(item, len(dsi_df[dsi_df["Tag"]==item]))
    # for t in list(set(pd.Series(tags))):
    #     print(t, tags[tags == t].count())
