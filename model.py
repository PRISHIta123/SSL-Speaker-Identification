import extract_mfcc as em
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from statistics import mean

files=os.listdir("C://Users/PrishitaRay/Desktop/SSL-Speaker-Identification/CHAINS/data/sync/8 speakers/sentences_audio_files")
mfcc_list=[]

os.chdir("C://Users/PrishitaRay/Desktop/SSL-Speaker-Identification/CHAINS/data/sync/8 speakers/sentences_audio_files")

for i in range(0,len(files)):
    sample_rate, signal= scipy.io.wavfile.read(files[i])

    #Select first five seconds of signal
    signal=signal[0:int(5*sample_rate)]

    #plt.plot(signal)
    #plt.xlabel("Time")
    #plt.ylabel("Amplitude")
    #plt.title("Speech signal plot")
    #plt.show()

    #plotting mfcc of speech signal
    mfcc= em.MFCC(signal)
    #print(mfcc)
    #plt.plot(mfcc)
    #plt.title("Plot of MFCC")
    #plt.show()
    mfcc_list.append(mfcc)

speakers=[]
n=33
speakers += 33*['frf01']
speakers += 33*['frf02']
speakers += 33*['frf03']
speakers += 33*['frf04']
speakers += 33*['frm01']
speakers += 33*['frm02']
speakers += 33*['frm03']
speakers += 33*['frm04']

speakers=np.array(speakers)

#Integer encoding speaker labels
le=LabelEncoder()
speaker_labels= np.array(le.fit_transform(speakers))
mfcc_list=list(mfcc_list)

#Dictionary to map integer encodings with speaker ids
l1= list(np.unique(speakers))
l2= list(np.unique(speaker_labels))
speaker_id_map= dict(zip(l2,l1))
print(speaker_id_map)

#Average of mfcc coefficients over entire signal length of each audio sample
mfcc_avg=[]

for m in mfcc_list:
    sm=0
    cnt=0
    for x in m:
        sm=sm+x
        cnt=cnt+1
    avg=sm/cnt
    mfcc_avg.append(avg)

#Dictionary to map mfccs with audio files
audio_file_map=dict(zip(files,preprocessing.normalize(list(mfcc_avg))))
        
data=[]
for i in range(0,len(mfcc_avg)):
    l=[]
    l.append(mfcc_avg[i])
    l.append(speaker_labels[i])
    data.append(l)
    
data=np.array(data)
np.random.shuffle(data)

#Initialize Classifiers
nb_clf = GaussianNB()
svm_clf = svm.SVC()
lr_clf = LogisticRegression(random_state=0)

#75% of the data is kept for training, 55% are labeled samples, 20% are unlabeled samples
#25% of the data is kept for testing
train_X_labeled=data[0:int(0.55*len(data)),0:1]
train_Y=data[0:int(0.55*len(data)),1]
train_X_unlabeled=data[int(0.55*len(data)):int(0.75*len(data)),0:1]
test_X=data[int(0.75*len(data)):len(data),0:1]
test_Y=data[int(0.75*len(data)):len(data),1]

labeled_X=[]
Y=train_Y.astype(int)
for i in range(0,len(train_X_labeled)):
    l=[]
    for x in train_X_labeled[i]:
        for y in x:
            l.append(y)
    labeled_X.append(l)

labeled_X=preprocessing.normalize(labeled_X)

unlabeled_X=[]
for i in range(0,len(train_X_unlabeled)):
    l=[]
    for x in train_X_unlabeled[i]:
        for y in x:
            l.append(y)
    unlabeled_X.append(l)

unlabeled_X=preprocessing.normalize(unlabeled_X)

test=[]
for i in range(0,len(test_X)):
    l=[]
    for x in test_X[i]:
        for y in x:
            l.append(y)
    test.append(l)

test=preprocessing.normalize(test)

cnt=0
cnt1=0
cnt2=0
cnt3=0
while len(unlabeled_X)!=0:
    #Calculate 10-fold cross validation score for each classifier
    print(cnt)
    cnt=cnt+1
    print(labeled_X.shape)
    nb_clf.fit(labeled_X,Y)
    scores1 = cross_val_score(nb_clf, labeled_X, Y, cv=10)
    val1= mean(scores1)
    
    svm_clf.fit(labeled_X,Y)
    scores2 = cross_val_score(svm_clf, labeled_X, Y, cv=10)
    val2= mean(scores2)
    
    lr_clf.fit(labeled_X,Y)
    scores3 = cross_val_score(lr_clf, labeled_X, Y, cv=10)
    val3= mean(scores3)
    y=[]

    #Choose the classifier with the highest mean cross validation score for the most confident prediction
    if max([val1,val2,val3])==val1:
        y=nb_clf.predict([unlabeled_X[0]])
        cnt1=cnt1+1

    elif max([val1,val2,val3])==val2:
        y=svm_clf.predict([unlabeled_X[0]])
        cnt2=cnt2+1
        
    elif max([val1,val2,val3])==val3:
        y=lr_clf.predict([unlabeled_X[0]])
        cnt3=cnt3+1

    #Iteratively add predicted values of unlabeled samples to labeled list
    labeled_X= np.vstack([labeled_X,unlabeled_X[0]])
    Y= np.append(Y,y[0])
    unlabeled_X= np.delete(unlabeled_X,0,0)       

y_pred=[]
#Predict speakers based on mfcc values of test audio samples with the most confident classifier
if max([cnt1,cnt2,cnt3])==cnt1:
    y_pred=nb_clf.predict(test)

elif max([cnt1,cnt2,cnt3])==cnt2:
    y_pred=svm_clf.predict(test)
    
elif max([cnt1,cnt2,cnt3])==cnt3:
    y_pred=lr_clf.predict(test)

pred_speakers=[]
for y in y_pred:
    pred_speakers.append(speaker_id_map[y])

test_files=[]
for t in test:
    for x in audio_file_map.keys():
        if np.array_equal(np.array(t),audio_file_map[x]):
            test_files.append(x)
            break
        
#Display predictions
preds=dict(zip(test_files,pred_speakers))

print("The predicted values are:\n")
for key in preds.keys():
    print(key,": speaker_id ",preds[key])
