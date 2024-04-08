import matplotlib.pyplot as plt
import numpy as np


def MakeGraphicPredictions(labels_train_true, train_pred_new, title='Previsões do Modelo', savepath='/prev.pdf'):
        plt.figure(figsize=(6,5))
        plt.scatter(labels_train_true, train_pred_new, s=8, alpha=0.6)
        plt.xlabel('Valores Reais [BPM]')
        plt.ylabel('Previsões [BPM]')
        plt.title (title)
        #plt.axis('equal')
        #plt.axis('square')
        plt.xlim([0,plt.xlim()[1]])
        plt.ylim([0,plt.ylim()[1]])
        plt.fill_between([0, 257], [0, (257*1.04)], [0, (257*0.96)], alpha=0.2, color='lightcoral')
        plt.fill_between([0, 257], [0, ((257/2)*1.04)], [0, ((257/2)*0.96)], alpha=0.2, color='mediumspringgreen')
        plt.fill_between([0, 257], [0, ((257*2)*1.04)], [0, ((257*2)*0.96)], alpha=0.2, color='mediumspringgreen')
        plt.fill_between([0, 257], [0, ((257/3)*1.04)], [0, ((257/3)*0.96)], alpha=0.2, color='mediumspringgreen')
        plt.fill_between([0, 257], [0, ((257*3)*1.04)], [0, ((257*3)*0.96)], alpha=0.2, color='mediumspringgreen')
        plt.plot([0, 257], [0, 257], color='red', label="Acurácia 0")
        plt.plot([0, 257], [0, 257/2], color='darkgreen', label="Acurácia 2")
        plt.plot([0, 257], [0, 257*2], color='darkgreen')
        plt.plot([0, 257], [0, 257*3], color='darkgreen')
        plt.plot([0, 257], [0, 257/3], color='darkgreen')
        plt.legend()
        plt.savefig(savepath, format='pdf')
        plt.show()
        
def MakeGraphicPredictions_english(labels_train_true, train_pred_new, title='Predictions', savepath='/prev.pdf'):
        plt.figure(figsize=(6,5))
        plt.scatter(labels_train_true, train_pred_new, s=8, alpha=0.6)
        plt.xlabel('True Labels [BPM]')
        plt.ylabel('Predictions [BPM]')
        plt.title (title)
        #plt.axis('equal')
        #plt.axis('square')
        plt.xlim([0,plt.xlim()[1]])
        plt.ylim([0,plt.ylim()[1]])
        plt.fill_between([0, 257], [0, (257*1.04)], [0, (257*0.96)], alpha=0.2, color='lightcoral')
        plt.fill_between([0, 257], [0, ((257/2)*1.04)], [0, ((257/2)*0.96)], alpha=0.2, color='mediumspringgreen')
        plt.fill_between([0, 257], [0, ((257*2)*1.04)], [0, ((257*2)*0.96)], alpha=0.2, color='mediumspringgreen')
        plt.fill_between([0, 257], [0, ((257/3)*1.04)], [0, ((257/3)*0.96)], alpha=0.2, color='mediumspringgreen')
        plt.fill_between([0, 257], [0, ((257*3)*1.04)], [0, ((257*3)*0.96)], alpha=0.2, color='mediumspringgreen')
        plt.plot([0, 257], [0, 257], color='red', label="Accuracy 0")
        plt.plot([0, 257], [0, 257/2], color='darkgreen', label="Accuracy 2")
        plt.plot([0, 257], [0, 257*2], color='darkgreen')
        plt.plot([0, 257], [0, 257*3], color='darkgreen')
        plt.plot([0, 257], [0, 257/3], color='darkgreen')
        plt.legend()
        plt.savefig(savepath, format='pdf')
        plt.show()
        
def acuracia2(labels,predictions):
    labelsr= np.around(labels)
    predictionsr= np.around(np.absolute(predictions))    
    TP0=0
    for i in range(labelsr.size):
        if (labelsr[i]-predictionsr[i])==0:
            TP0=TP0+1
    acuracia0=TP0/labelsr.size
    TP1=0
    #print('Acurácia 0 = ',acuracia0) 
    for i in range(labelsr.size):
        #print('posição i:',i)
        if (np.absolute(labelsr[i]-predictionsr[i]))<=(labelsr[i]*0.04):
            TP1=TP1+1
            #print('labelsr[i]-predictionsr[i]:',np.absolute(labelsr[i]-predictionsr[i]))
            #print('4%:',labelsr[i]*0.04)
            #print('TP:',TP1)            
    acuracia1=TP1/labelsr.size
    #print('Acurácia 1 = ',acuracia1)
    TP2=0
    predictionsr2=np.around(predictions)
    predictionsr3=np.around(predictions)
    for i in range(labelsr.size):
        #print('label-prediction:',(np.absolute(labelsr[i]-predictionsr[i])))
        #print('4%:',(labelsr[i]*0.04))
        #print('calculo 2:',(np.absolute(np.around(labelsr[i]/2)-predictionsr[i])))
        #print('calculo 3:',(np.absolute(np.around(labelsr[i]*2)-predictionsr[i])))
        #print('calculo 4:',(np.absolute(np.around(labelsr[i]/3)-predictionsr[i])))
        #print('calculo 5:',(np.absolute(np.around(labelsr[i]*3)-predictionsr[i])))        
        if (np.absolute(labelsr[i]-predictionsr[i]))<=(labelsr[i]*0.04):
            TP2=TP2+1
        #    print('condição 1')
        elif (np.absolute(np.around(labelsr[i]/2)-predictionsr[i]))<=(labelsr[i]*(0.04/2)):
            TP2=TP2+1
        #    print('condição 2')
        elif (np.absolute(np.around(labelsr[i]*2)-predictionsr[i]))<=(labelsr[i]*(0.04*2)):
            TP2=TP2+1
        #    print('condição 3')
        elif (np.absolute(np.around(labelsr[i]/3)-predictionsr[i]))<=(labelsr[i]*(0.04/3)):
            TP2=TP2+1
        #    print('condição 4')
        elif (np.absolute(np.around(labelsr[i]*3)-predictionsr[i]))<=(labelsr[i]*(0.04*3)):
            TP2=TP2+1
         #   print('condição 5')
       # else:
         #   print('sem condição')       
    acuracia2=TP2/labelsr.size
    return acuracia2

def acuracia0(labels,predictions):
    labelsr= np.around(labels)
    predictionsr= np.around(np.absolute(predictions))    
    TP0=0
    for i in range(labelsr.size):
        if (labelsr[i]-predictionsr[i])==0:
            TP0=TP0+1
    acuracia0=TP0/labelsr.size
    TP1=0
    return acuracia0

def acuracia1(labels,predictions):
    labelsr= np.around(labels)
    predictionsr= np.around(np.absolute(predictions))   
    TP1=0
    for i in range(labelsr.size):
        #print('posição i:',i)
        if (np.absolute(labelsr[i]-predictionsr[i]))<=(labelsr[i]*0.04):
            TP1=TP1+1
            #print('labelsr[i]-predictionsr[i]:',np.absolute(labelsr[i]-predictionsr[i]))
            #print('4%:',labelsr[i]*0.04)
            #print('TP:',TP1)            
    acuracia1=TP1/labelsr.size   
    return acuracia1


