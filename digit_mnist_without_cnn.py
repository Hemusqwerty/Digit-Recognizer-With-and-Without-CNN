import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from struct import unpack
from mlxtend.data import loadlocal_mnist
train_img,train_lbl=loadlocal_mnist('train-images.idx3-ubyte',
                              'train-labels.idx1-ubyte')
test_img,test_lbl=loadlocal_mnist('t10k-images.idx3-ubyte',
                            't10k-labels.idx1-ubyte')
print(train_img.shape) #60000 img,28*28
print(train_lbl.shape)
print(test_img.shape)
print(test_lbl.shape)

plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

from sklearn.linear_model import LogisticRegression
logisticReg=LogisticRegression(solver='lbfgs') #load dataset faster
logisticReg.fit(train_img,train_lbl)
logisticReg.predict(test_img[0].reshape(1,-1))#predict for 1 img
logisticReg.predict(test_img[0:10]) #predict for 10 img
predictions=logisticReg.predict(test_img) #predict for all
predictions.shape
score=logisticReg.score(test_img,test_lbl)
print(score)

#confusion matrix
def plot_confusion_matrix(cm,title='Confusion Matrix',cmap='Pastel1'):
    plt.figure(figsize=(9,9))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title,size=15)
    plt.colorbar()
    tick_marks=np.arange(10)
    plt.xticks(tick_marks,["0","1","2","3","4","5","6","7","8","9"],rotation=45,size=10)
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
    plt.tight_layout()
    plt.ylabel("Actual label",size=15)
    plt.xlabel("Prediction label",size=15)
    width,height=cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]),xy=(y,x),
            horizontalalignment='center',
            verticalalignment='center')
confusion=metrics.confusion_matrix(test_lbl,predictions)
print("Confusion Matrix")
print(confusion)
plot_confusion_matrix(confusion)
plt.show()
#Seaborn Confusion Matrix
cm=metrics.confusion_matrix(test_lbl,predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt=".3f",linewidths=.5,square=True,cmap='Blues_r')
all_sample_title='Accuracy Score:{0}'.format(score)
plt.title(all_sample_title,size=15)
plt.show()

index=0
misclassifiedIndex=[]
for predict,actual in zip(predictions,test_lbl):
    if predict!=actual:
        misclassifiedIndex.append(index)
    index+=1
plt.figure(figsize=(20,4))
for plotIndex,wrong in enumerate(misclassifiedIndex[10:15]):
    plt.subplot(1,5,plotIndex+1)
    plt.imshow(np.reshape(test_img[wrong],(28,28)),cmap=plt.cm.gray)
    plt.title('Predicted: {},Actual: {}'.format(predictions[wrong],test_lbl[wrong]),fontsize=14)
plt.show()

