import re
import pandas as pd
import jieba as jb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer  # 词频计数
from sklearn.feature_extraction.text import TfidfVectorizer  # tf-idf 模块
from sklearn.model_selection import train_test_split
jb.load_userdict('tools/自定义词典.txt')


path = r'D:\Users\Desktop\train2022228.csv'
df = pd.read_csv(path)
df.columns = ["lable", "data"]


# 停用词列表
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 加载停用词
stopwords = stopwordslist("tools/stop_words")




corpus = []
for i in df['data']:
    cor = jb.lcut(i)
    # print(i)
    outstr = ''
    for j in cor:
        if j not in stopwords:
            if j != '\t':
                outstr += j
                outstr += " "
                outstr = outstr.replace(" ", "|")
    corpus.append(outstr)

vectorizer = CountVectorizer()
corpusTotoken_count = vectorizer.fit_transform(corpus).todense()
vectorizer = TfidfVectorizer()
corpusTotoken_tfidf = vectorizer.fit_transform(corpus).todense()


X_data = np.array(corpusTotoken_count)
Y_data = np.array(df['lable'])

x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)

# LR预测
LR = LogisticRegression()
LR.fit(x_train, y_train)
predictions_LR = LR.predict(x_test)
prob_LR = LR.predict_proba(x_test)
LR_y_predicted = LR.predict(x_train)
# print(metrics.classification_report(y_train, LR_y_predicted))
LR_predicted = np.mean(LR_y_predicted == y_train)
print('LR准确率：', LR_predicted)

# Bernoulli bayes 预测
GB = MultinomialNB()
GB.fit(x_train, y_train)
predictions_GB = GB.predict(x_test)
prob_GB = GB.predict_proba(x_test)
# print('predictions_GB:', predictions_GB)

# 模型评估
GB_y_predicted = GB.predict(x_train)
# print(metrics.classification_report(y_train, GB_y_predicted))
GB_predicted = np.mean(GB_y_predicted == y_train)
print('bayes准确率：', GB_predicted)

# RandomForest预测
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
predictions_RF = RF.predict(x_test)
prob_RF = RF.predict_proba(x_test)
# print('RandomForestClassifier:', prob_RF)

# 模型评估
RF_y_predicted = RF.predict(x_train)
# print(metrics.classification_report(y_train, GB_y_predicted))
RF_predicted = np.mean(RF_y_predicted == y_train)
print('RandomForest准确率：', RF_predicted)


# SVM预测
SV = SVC(kernel='linear', probability=True)
SV.fit(x_train, y_train)
predictions_SV = SV.predict(x_test)
prob_SV = SV.predict_proba(x_test)

# 模型评估
SV_y_predicted = SV.predict(x_train)
# print(metrics.classification_report(y_train, GB_y_predicted))

SV_predicted = np.mean(SV_y_predicted == y_train)
print('SVM准确率：', SV_predicted)

while 1:
    sent = input("客户问：")
    sent_cut = jb.lcut(sent)
    sent_cut_input = ' '.join(sent_cut)
    print("切词结果：", sent_cut_input)
    Xpredict = vectorizer.transform([sent_cut_input]).todense()
    result = GB.predict_proba(Xpredict)[0]
    print("预测的准确率=", f'问的概率：{result[1]} 答的概率：{result[0]}')

    prediction = GB.predict(Xpredict)[0]
    print('预测标签：', prediction)
    print('*'*100)
