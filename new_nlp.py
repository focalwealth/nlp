# Model Training

from sklearn.feature_extraction.text import TfidfVectorizer as tfidf, HashingVectorizer as hv, CountVectorizer as CV
from nltk.stem.porter import *
import nltk
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score, classification_report
import json
import pandas as pd
from collections import defaultdict
import pickle
from pandas.io.json import json_normalize
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def tokenizer(data_entry):
    token = word_tokenize(data_entry)
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in token]
    stopwords = nltk.corpus.stopwords.words('portuguese')
    # generate stopwords without accents
    stop_symbols = '{}()[].,:;+-*/&!@#$%^_\|%<>=~$1234567890'
    non_sw_list = []
    for words in stemmed:
        if words not in stopwords:
            if any(x in stop_symbols for x in words):
                pass
                # print words, "rejected"
            else:
                non_sw_list.append(words)

    return non_sw_list


def cleaner(df):
    try:
        data_list = df["Data"].tolist()
    except:
        data_list = df["Data"]
    try:
        tag_list = df["Tag"].tolist()
    except:
        tag_list = df["Tag"]
    ps = PorterStemmer()
    new = []
    for data in data_list:
        data = data.lower()
        data = tokenizer(data)
        data = [ps.stem(word) for word in data]
        data = " ".join(data)
        new.append(data)

    return new, tag_list


def json_exploder(df):
    chats = []
    for ind in range(0, len(df["tnscpt_array"])):
        chat = []

        try:
            j = unicode(df["tnscpt_array"][ind], errors='replace')
            j = j.encode('utf-8')
        except:
            j = df["tnscpt_array"][ind]
            j = j.encode('utf-8')
        d = json.loads(str(j))
        for m in range(0, len(d)):
            chat.append(d[m]["text"])
        chats.append("\n".join(chat))
    df["tnscpt_array"] = chats
    return df


def tnscpt_json_exploder(df, cco=True, cl=0):
    if cco:
        print
        "Considering only customer chat as the input to model"
        chats = []
        if cl == 0:
            print
            "Using all the chats by customers"
            for ind in range(0, len(df["tnscpt_array"])):
                chat = []

                try:
                    j = json.loads(str(unicode(df["tnscpt_array"][ind], errors='replace').encode('utf-8')))
                except:
                    j = json.loads(transcript[ind].encode('utf-8'))
                ndf = json_normalize(j)
                cust = ndf[ndf["source"] == "visitor"]
                txt = " ".join(cust["text"].tolist())
                chats.append(txt)
        else:
            print
            "Using the top {} chats by customers".format(cl)
            for ind in range(0, len(df["tnscpt_array"])):
                chat = []

                try:
                    j = json.loads(str(unicode(df["tnscpt_array"][ind], errors='replace').encode('utf-8')))
                except:
                    j = json.loads(transcript[ind].encode('utf-8'))
                ndf = json_normalize(j)
                cust = ndf[ndf["source"] == "visitor"][:cl]
                txt = " ".join(cust["text"].tolist())
                chats.append(txt)
        df["tnscpt_array"] = chats
        return df

    else:
        print
        "Considering all chat data as the input to model"
        chats = []
        # print len(df["tnscpt_array"])
        for ind in range(0, len(df["tnscpt_array"])):
            chat = []

            try:
                j = unicode(df["tnscpt_array"][ind], errors='replace')
                j = j.encode('utf-8')
            except:
                j = df["tnscpt_array"][ind]
                j = j.encode('utf-8')
            d = json.loads(str(j))
            for m in range(0, len(d)):
                chat.append(d[m]["text"])
            chats.append("\n".join(chat))
        df["tnscpt_array"] = chats
        return df


def phrase_extraction_tfidf(dtm, features, clean_text, tags, top_n):
    print
    "Activating tfidf Phrase Extractor analysis:"
    print
    "Total number of features are:", len(features)
    indices = np.argsort(tf_idf.idf_)[::-1]
    features = tf_idf.get_feature_names()
    top = top_n
    top_features = [features[i] for i in indices[:top]]
    print
    "\nHere are the top {} important features:".format(top)
    print
    top_features


def run_random_forest_classifier(n, X_train, X_test, y_train, y_test, dump_pkl=False):
    print("\nRandom forest Classification:")
    clfrf = RandomForestClassifier(n_estimators=200, n_jobs=-1, bootstrap=True)
    clfrf.fit(X_train, y_train)
    prediction = clfrf.predict(X_test)
    print("Model Accuracy for cross validation - ", np.mean(cross_val_score(clfrf, X_test, y_test, cv=10)))
    print("Accuracy of the model on testing data: {}% ".format(accuracy_score(y_test, prediction) * 100))
    print("F1 score: {}".format(f1_score(y_true=y_test, y_pred=prediction, average='binary', pos_label=None)))
    # print ("Presicion Score is :{}".format(average_precision_score(y_test,prediction, average=)))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))

    if dump_pkl:
        filename = 'rf_model.pkl'
        print
        "Dumping Model {} to the current working directory".format(filename)
        pickle.dump(clfrf, open(filename, 'wb'))


def run_nb_classifier(X_train, X_test, y_train, y_test, dump_pkl=False):
    print("\nNaive Bayes Classification:")
    clfnb = GaussianNB()
    clfnb.fit(X_train, y_train)
    prediction = clfnb.predict(X_test)
    #print("Model Accuracy for cross validation - ", np.mean(cross_val_score(clfnb, X_test, y_test, cv=10)))
    print("Accuracy of the model on testing data: {}% ".format(accuracy_score(y_test, prediction) * 100))
    print("F1 score: {}".format(f1_score(y_true=y_test, y_pred=prediction, average='binary', pos_label=None)))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    if dump_pkl:
        filename = 'nb_model.pkl'
        print
        "Dumping Model {} to the current working directory".format(filename)
        pickle.dump(clfnb, open(filename, 'wb'))


def run_adaboost_classifier(n, X_train, X_test, y_train, y_test, dump_pkl=False):
    print("\nAdaBoost Classification:")
    clfab = AdaBoostClassifier(n_estimators=n)
    clfab.fit(X_train, y_train)
    prediction = clfab.predict(X_test)
    print("Model Accuracy for cross validation - ", np.mean(cross_val_score(clfab, X_test, y_test, cv=10)))
    print("Accuracy of the model on testing data: {}% ".format(accuracy_score(y_test, prediction) * 100))
    print("F1 score: {}".format(f1_score(y_true=y_test, y_pred=prediction, average='binary', pos_label=None)))
    # print ("Presicion Score is :{}".format(average_precision_score(y_test,prediction, average=)))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))


def run_GBoost_classifier(n, X_train, X_test, y_train, y_test, new_test, dump_pkl=False):
    print("\nGradient Boosting Classification:")
    clfgb = GradientBoostingClassifier(n_estimators=n)
    clfgb.fit(X_train, y_train)
    prediction = clfgb.predict(X_test)
    pred=clfgb.predict(new_test)
    # print("Model Accuracy for cross validation - ", np.mean(cross_val_score(clfgb, X_test, y_test, cv=2)))
    print("Accuracy of the model on testing data: {}% ".format(accuracy_score(y_test, prediction) * 100))
    # print("F1 score: {}".format(f1_score(y_true=y_test, y_pred=prediction, ave pos_label=None)))
    # print ("Presicion Score is :{}".format(average_precision_score(y_test,prediction, average=)))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("The prediction is:")
    print(pred)


def clustering(clean_text, tags, top_n, model='None', clusters=10):
    x, features = activate_tf_idf(clean_text, tags, top_n, model)
    print
    "Matrix Size= ", x.shape
    kmeans = KMeans(n_clusters=clusters)
    a = kmeans.fit(x)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    # print(centroids)
    # print(labels)
    Y_pred = kmeans.predict(x)
    print
    set(Y_pred)
    print
    len(Y_pred)
    plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=Y_pred, cmap=plt.cm.Paired)
    plt.legend()
    plt.title('train data')
    plt.show()


def activate_tf_idf(clean_text, tags, top_n, model="All", n_estimator=2000, mindf=0.001, ng=1):
    print(model)
    dft = pd.DataFrame({"Data": [txt], "Tag": ['M']})
    clean_text_test, tags_test = cleaner(dft)
    if model == 'None':
        print("\nActivating tfidf analysis for unsupervised learning:")
        tf_idf = tfidf(sublinear_tf=True, min_df=mindf, ngram_range=(0, ng))
        X = tf_idf.fit_transform(clean_text)
        dtm = pd.DataFrame(X.toarray())
        features = tf_idf.get_feature_names()
        return dtm, features
    else:
        print("\nActivating tfidf analysis:")
        tf_idf = tfidf(sublinear_tf=True, min_df=mindf, ngram_range=(0, ng))
        X = tf_idf.fit_transform(clean_text)
        dtm = pd.DataFrame(X.toarray())
        features = tf_idf.get_feature_names()
        print ("Total number of features are:", len(features))
        # phrase_extraction_tfidf(dtm, features, clean_text, tags, top_n, tf_idf=tf_idf)
        if top_n:
            indices = np.argsort(tf_idf.idf_)[::-1]
            features = tf_idf.get_feature_names()
            top = top_n
            top_features = [features[i] for i in indices[:top]]
            print("\nHere are the top {} important features:".format(top))
            print(top_features)
        X_train, X_test, y_train, y_test = train_test_split(dtm, tags, test_size=0.05, random_state=40)
        X_test = dtm.iloc[0:31, :]
        y_test = tags[0:31]
        new_test = tf_idf.transform(clean_text_test)
        model_runner(model, n_estimator, X_train, X_test, y_train, y_test, new_test)


def hashing_vector(clean_text, tags, model="All"):
    print
    "\nActivating HV analysis:"
    hv_vec = hv(stop_words="english")
    X = hv_vec.fit_transform(clean_text)
    dtm = pd.DataFrame(X.toarray())
    # features=hv_vec.get_feature_names()
    # print len(features)
    # print features[0:100]
    X_train, X_test, y_train, y_test = train_test_split(dtm, tags, test_size=0.05, random_state=40)
    model_runner(model, X_train, X_test, y_train, y_test)


def model_runner(model, n_estimator, X_train, X_test, y_train, y_test, new_test):
    if model == "All":
        print
        "\nRunning all the Models:"
        run_adaboost_classifier(n_estimator, X_train, X_test, y_train, y_test)
        run_random_forest_classifier(n_estimator, X_train, X_test, y_train, y_test)
        run_GBoost_classifier(n_estimator, X_train, X_test, y_train, y_test)
        run_nb_classifier(X_train, X_test, y_train, y_test)
    elif model == "GBoost":
        print("Gbooster run")
        run_GBoost_classifier(n_estimator, X_train, X_test, y_train, y_test, new_test)
    elif model == "RandomForest":
        run_random_forest_classifier(n_estimator, X_train, X_test, y_train, y_test)
    elif model == "AdaBoost":
        run_adaboost_classifier(n_estimator, X_train, X_test, y_train, y_test)
    else:
        print
        "Error passing the model execution. Please pass the correct Parameter to the function: model_runner"


def pre_executer(model='All', n_estimator=2000, mindf=0.001, ng=1, cust_chat_only=True, chats_len=0, top_n=False,
                 cluster=False, clusters=10):
    print('Analysing the post submit data')
    print("Parameters Passed:")
    print("n_estimator =", n_estimator)
    print("min_df =", mindf)
    print("n_grams =", ng)
    print(cluster)
    if cluster:
        clustering(clean_text, tags, top_n, model, clusters)
    else:
        print("Activating")
        activate_tf_idf(clean_text, tags, top_n, model=model, n_estimator=n_estimator, mindf=mindf, ng=ng)


def post_executer(model='All', n_estimator=2000, mindf=0.001, ng=1, cust_chat_only=True, chats_len=0, top_n=False,
                  cluster=False, clusters=10):
    print('Analysing the post submit data')
    print("Parameters Passed:")
    print("n_estimator =", n_estimator)
    print("min_df =", mindf)
    print("n_grams =", ng)
    df_chatdata = pd.read_csv("SQLAExport.txt", sep="\t")
    chat_df = tnscpt_json_exploder(df_chatdata, cco=cust_chat_only)  # By default cust_only_chat=True
    df_post = pd.read_csv("post_chat_tagged.txt", sep="\t")
    for i in range(0, len(df_post["Engagement_ID"])):
        # print final["Engagement_ID"][i]
        df_post["Engagement_ID"][i] = "3824612" + str(df_post["Engagement_ID"][i])

    final = chat_df.join(df_post.set_index('Engagement_ID'), on='engagment_id')
    print(len(final))
    tags = []

    for i in range(0, len(final["segment"])):
        #         print "executing", i
        try:
            if final["CA"][i] == "CA":
                #                 print "entered ca", i
                tags.append(2)
            elif final["D"][i] == "Q":
                #                 print "entered q", i
                tags.append(1)
            else:
                #                 print "entered nan" ,i
                tags.append(0)
        except:
            #             print "executing", i
            tags.append(0)
    passer = pd.DataFrame({"Data": final["tnscpt_array"].tolist(), "Tag": tags})
    #clean_text, tags = cleaner(passer)
    # hashing_vector(clean_text, tags)
    if cluster:
        clustering(clean_text, tags, top_n, model, clusters)
    else:
        activate_tf_idf(clean_text, tags, top_n, model=model, n_estimator=n_estimator, mindf=mindf, ng=ng)


if __name__ == "__main__":
    #     pre_executer(model= 'Gboost', n_estimator=200, mindf=0.01, ng=3, cust_chat_only=True, chats_len=10, top_n= 150)
    #     post_executer(model= 'None', n_estimator=2000, mindf=0.001, ng=2, cust_chat_only=False, chats_len=0, top_n= 150)
    # Best model config
    # pre_executer(model='GBoost', n_estimator=2000, mindf=0.001, ng=1, cust_chat_only=True, chats_len=0,
    #              top_n=150)  # gives 87.5
    df = pd.read_csv("C:\\Users\Sheel\PycharmProjects\\NLP\Data_Set_new.txt", sep="\t", encoding='latin-1', header=None)
    dsi = {"URL": df[0].tolist(), "Tag": df[1].tolist(), "Data": df[2].tolist()}
    dsi_df = pd.DataFrame(dsi)
    print(dsi_df.shape)
    word_set = []
    clean_text, tags = cleaner(dsi_df)
    print(len(tags))
    txt = """"test here"""
    pre_executer(model='GBoost', n_estimator=2000, mindf=0.001, ng=3, cust_chat_only=True, chats_len=10, top_n=150)