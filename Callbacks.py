import keras
import matplotlib.pyplot as plt
import numpy as np
from Utils import Algo
from sklearn.metrics import roc_curve, auc

class Plot_test(keras.callbacks.Callback):
    def __init__(self, test_data, folder):
        self.test_data = test_data
        self.folder = folder
        self.epochs = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epochs += 1

        if self.epochs % 10 == 0:
                steps = [99, 4, 23, 22, 5, 24, 7, 6, 8, 9, 10, 11]
                test_x, test_y = (self.test_data[0].copy(), self.test_data[1].copy())
                x_ = self.test_data
                Nentries=x_.shape[0]
                pred_ = self.model.predict(test_x.drop(['trk_isTrue', 'trk_mva'], axis=1))
                x_['trk_mva_DNN'] = pred_

                ind=0
                for step in steps:
                        if step==99:
                                data_pl=x_
                                y_=test_y
                        else:
                                indices_ = (x_.trk_algo==step)
                                data_pl=x_[indices_]
                                y_=test_y[indices_]

                        true_trks=data_pl[(y_==1)]
                        fake_trks=data_pl[(y_==0)]

                        frac_true=np.round(1.0*true_trks.shape[0]/Nentries,3)
                        frac_fake=np.round(1.0*fake_trks.shape[0]/Nentries,3)

                        true_trks['trk_mva_DNN']=true_trks.loc[:,'trk_mva_DNN'].apply(lambda row: 2.0*row-1.0)
                        fake_trks['trk_mva_DNN']=fake_trks.loc[:,'trk_mva_DNN'].apply(lambda row: 2.0*row-1.0)


                        #Distr of mva output
                        plt.hist(true_trks['trk_mva_DNN'],bins=np.linspace(-1.,1.,20),label='True tracks',density=True,alpha=0.7)
                        plt.hist(fake_trks['trk_mva_DNN'],bins=np.linspace(-1.,1.,20),label='Fake tracks',density=True,alpha=0.7)
                        plt.legend()
                        plt.title('MVA distribution for ' + Algo.toString(step)+', fraction = '+str(frac_true+frac_fake))
                        plt.xlabel('MVA value')
                        plt.ylabel('Number of tracks per bin')
                        plt.savefig('plots_MVADist_'+self.folder+'/'+Algo.toString(step)+'_MVA_distribution_epoch_'+str(self.epochs)+'.pdf')
                        plt.clf()

                        #mva vs. pT
                        #True tracks
                        binning_pt=np.logspace(-1.0,3.0,num=40)
                        indices_true = np.digitize(true_trks.trk_pt,binning_pt)

                        weights_DNN = np.array([np.mean(true_trks[indices_true==x].trk_mva_DNN) for x in range(40)])
                        err_DNN = np.array([np.std(true_trks[indices_true==x].trk_mva_DNN) for x in range(40)])
                        err_DNN[np.isnan(err_DNN)]= 1
                        err_DNN[err_DNN==0]= 1
                        weights_MVA = np.array([np.mean(true_trks[indices_true==x].trk_mva) for x in range(40)])
                        err_MVA = np.array([np.std(true_trks[indices_true==x].trk_mva) for x in range(40)])
                        err_MVA[np.isnan(err_MVA)]= 1
                        err_MVA[err_MVA==0]= 1

                        plt.scatter(binning_pt,weights_DNN,label='DNN',color='blue')
                        plt.fill_between(binning_pt,weights_DNN-err_DNN,weights_DNN+err_DNN,alpha=0.4,label='$\pm 1\sigma$',color='blue')
                        plt.scatter(binning_pt,weights_MVA,label='BDT',color='red')
                        plt.fill_between(binning_pt,weights_MVA-err_MVA,weights_MVA+err_MVA,alpha=0.4,label='$\pm 1\sigma$',color='red')
                        plt.xscale('log')
                        plt.ylim(-1.0,1.0)
                        plt.title('Average MVA output for true tracks, fraction = '+str(frac_true))
                        plt.ylabel('MVA output')
                        plt.xlabel('p_T (GeV)')
                        plt.legend()
                        plt.grid()
                        plt.savefig('plots_ROCs_'+self.folder+'/'+Algo.toString(step)+'_MVA_true_epoch_'+str(self.epochs)+'.pdf')
                        plt.clf()

                        #Fake tracks

                        indices_fake = np.digitize(fake_trks.trk_pt,binning_pt)

                        weights_DNN = np.array([np.mean(fake_trks[indices_fake==x].trk_mva_DNN) for x in range(40)])
                        err_DNN = np.array([np.std(fake_trks[indices_fake==x].trk_mva_DNN) for x in range(40)])
                        err_DNN[np.isnan(err_DNN)]= 1
                        err_MVA[err_MVA==0] = 1
                        weights_MVA = np.array([np.mean(fake_trks[indices_fake==x].trk_mva) for x in range(40)])
                        err_MVA = np.array([np.std(fake_trks[indices_fake==x].trk_mva) for x in range(40)])
                        err_MVA[np.isnan(err_MVA)] = 1
                        err_MVA[err_MVA==0] = 1

                        plt.scatter(binning_pt,weights_DNN,label='DNN',color='blue')
                        plt.fill_between(binning_pt,weights_DNN-err_DNN,weights_DNN+err_DNN,alpha=0.4,label='$\pm 1\sigma$',color='blue')
                        plt.scatter(binning_pt,weights_MVA,label='BDT',color='red')
                        plt.fill_between(binning_pt,weights_MVA-err_MVA,weights_MVA+err_MVA,alpha=0.4,label='$\pm 1\sigma$',color='red')
                        plt.xscale('log')
                        plt.ylim(-1.0,1.0)
                        plt.title('Average MVA output for fake tracks, fraction= '+str(frac_fake))
                        plt.ylabel('MVA output')
                        plt.xlabel('p_T (GeV)')
                        plt.legend()
                        plt.grid()
                        plt.savefig('plots_ROCs_'+self.folder+'/'+Algo.toString(step)+'_MVA_fake_epoch_'+str(self.epochs)+'.pdf')
                        plt.clf()

                        #ROC Curve

                        fpr_BDT, tpr_BDT, thresholds = roc_curve(data_pl['trk_isTrue'],data_pl['trk_mva'])
                        frr_BDT = np.ones(len(fpr_BDT))-fpr_BDT
                        AUC_BDT=auc(frr_BDT,tpr_BDT)
                        fpr_DNN, tpr_DNN, thresholds = roc_curve(data_pl['trk_isTrue'],data_pl['trk_mva_DNN'])
                        frr_DNN = np.ones(len(fpr_DNN))-fpr_DNN
                        AUC_DNN=auc(frr_DNN,tpr_DNN)

                        plt.plot(tpr_DNN,frr_DNN,label='DNN, AUC '+str(round(AUC_DNN,3)),color='blue')
                        plt.plot(tpr_BDT,frr_BDT,label='BDT, AUC '+str(round(AUC_BDT,3)),color='red')
                        plt.title(' ROC curve')
                        plt.ylabel('Fake rejection rate')
                        plt.xlabel('True selection rate')
                        plt.ylim(0.0,1.1)
                        plt.xlim(0.0,1.1)
                        plt.legend()
                        plt.grid(color='black',ls='--')
                        plt.savefig('plots_ROCs_'+self.folder+'/'+Algo.toString(step)+'_ROC_curve_epoch_'+str(self.epochs)+'.pdf')
                        plt.clf()


