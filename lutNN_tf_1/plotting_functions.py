import tensorflow as tf
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import colors
#from sklearn.metrics import plot_confusion_matrix
#from sklearn.metrics import ConfusionMatrixDisplay
#from sklearn.metrics import roc_curve
#from functools import partial

###################################################
###################################################
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (14, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         #'xticks':'major_ticks_top'
         }

plt.rcParams.update(params)
###################################################
###################################################
cumulativePosteriorCut = 0.7

ptCuts = (10, 15, 20, 25, 30, 40)
qualityCut =  12
###################################################
###################################################
def plotEvent(features, labels):
    image = features 
    pt = labels[0][0]
    charge = None
    image = plt.matshow(image, aspect = 5.0, origin = "lower", cmap="gnuplot2")
    plt.title("Gen: pt: {:+.1f}, charge: {}".format(pt,charge))

###################################################
###################################################
def plotPosterior(ptGen, labels, predictions, label2Pt, testIndex=0):
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
            
    if testIndex==0:
        indices = np.logical_and(labels>ptGen-0.1, labels<ptGen+0.1)
        predictions = predictions[indices]
        
    predictions = predictions[testIndex] 
    ###
    #x = np.roll(predictions, 1, axis=0)
    #predictions -=x
    ###
    
    predictions = tf.reshape(predictions, (1,-1))    
    predictions = np.mean(predictions, axis=0)
    maxPosterior = tf.math.reduce_max(predictions)
    scaleFactor = int(0.8/maxPosterior + 0.5)
    axes[0].plot(label2Pt(np.arange(predictions.shape[0])), scaleFactor*predictions, label="{}xposterior".format(scaleFactor))
    axes[0].plot(label2Pt(np.arange(predictions.shape[0])), np.cumsum(predictions), linestyle='-.',label="cumulative posterior")
    axes[1].plot(label2Pt(np.arange(predictions.shape[0])), scaleFactor*predictions, label="{}xposterior".format(scaleFactor))
    
    predictions = np.cumsum(predictions, axis=0)>cumulativePosteriorCut
    predictions = np.argmax(predictions, axis=0)
    ptRec = label2Pt(predictions)
    print("Pt gen = {:+.1f}, Pt rec {} cumulative posterior cut: {}".format(ptGen, ptRec, cumulativePosteriorCut))
    axes[0].axvline(ptGen, linestyle='-', color="olivedrab", label=r'$p_{T}^{GEN} \pm 1 [GeV/c]$')
    
    axes[0].set_xlabel(r'$p_{T} [GeV/c]$')
    axes[0].set_ylabel('Value')
    axes[0].set_xlim([0, 2*ptGen])
    axes[0].set_ylim([1E-3,1.05])
    
    axes[0].legend(bbox_to_anchor=(2.5,1), loc='upper left')
    axes[1].set_xlabel(r'$p_{T} [GeV/c]$')
    axes[1].set_ylabel('Value')
    axes[1].set_xlim([0,201])
    #axes[1].set_ylim([1E-3,1.05])
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.3)
    plt.savefig("fig_png/Posterior_ptGen_{}.png".format(ptGen), bbox_inches="tight")
###################################################
###################################################    
def plotTurnOn(df, ptCut, qualityCut, plotDir):
    #ptMax = ptCut+30
    #nPtBins = int(ptMax*2.0)
    ptMax = 200
    nPtBins = int(ptMax)
    
    ptHistoBins = range(0,nPtBins+1)
    
    denominator, ptHistoBins1 = np.histogram(df["GEN_pt"], bins=ptHistoBins) 
    numerator_OMTF, _ = np.histogram(df[df["OMTF_pt"]>=ptCut]["GEN_pt"], bins=ptHistoBins)
    #numerator_NN, _ = np.histogram(df[(df["NN_pt"]>=ptCut)&(df["NN_prob"]>0.07)]["GEN_pt"], bins=ptHistoBins)
    numerator_NN, _ = np.histogram(df[(df["NN_pt"]>=ptCut) & (df["OMTF_quality"]>=qualityCut)]["GEN_pt"], bins=ptHistoBins)
    
    #print("plotTurnOn, ptCut", ptCut, " ptMax ", ptMax, " nPtBins ", nPtBins)   
    #print("ptHistoBins ", ptHistoBins)
    #print("ptHistoBins1", ptHistoBins1)
       
    ratio_OMTF = np.divide(numerator_OMTF, denominator, out=np.zeros(denominator.shape), where=denominator>0)
    ratio_NN = np.divide(numerator_NN, denominator, out=np.zeros(denominator.shape), where=denominator>0)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        #y, x
    axes[0, 0].plot(ptHistoBins[:-1],numerator_OMTF, "ro", label="OMTF",linewidth=2)
    axes[0, 0].plot(ptHistoBins[:-1],numerator_NN, "bo", label="NN pt >= " + str(ptCut) + " q >= " + str(qualityCut), linewidth=2)
    axes[0, 0].set_xlim([0,2.0*ptCut])
    axes[0, 0].set_xlabel(r'$p_{T}^{GEN}$')
    axes[0, 0].set_ylabel('Events passing pT cut')
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].grid(visible=True, which='both')
    
    axes[1, 0].plot(ptHistoBins[:-1],numerator_OMTF, "ro", label="OMTF",linewidth=2)
    axes[1, 0].plot(ptHistoBins[:-1],numerator_NN, "bo", label="NN pt >= " + str(ptCut) + " q >= " + str(qualityCut), linewidth=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlim([0,2.0*ptCut])
    axes[1, 0].set_xlabel(r'$p_{T}^{GEN}$')
    axes[1, 0].set_ylabel('Events passing pT cut')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(visible=True, which='both')
    
    axes[1, 1].plot(ptHistoBins[:-1],ratio_OMTF, "ro", label="OMTF", linewidth=2)
    axes[1, 1].plot(ptHistoBins[:-1],ratio_NN, "bo", label="NN pt >= " + str(ptCut) + " q >= " + str(qualityCut))
    axes[1, 1].grid(visible=True, which='both')
    axes[1, 1].tick_params(which='both')
    #axes[1, 1].set_yticks(which='both')
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlim([0,2.0*ptCut])
    axes[1, 1].set_ylim([1E-3,1.05])
    axes[1, 1].set_xlabel(r'$p_{T}^{GEN}$')
    axes[1, 1].set_ylabel('Efficiency')
    
    axes[0, 1].plot(ptHistoBins[:-1],ratio_OMTF, "r",label="OMTF")
    axes[0, 1].plot(ptHistoBins[:-1],ratio_NN, "b", label="NN pt >= " + str(ptCut) + " q >= " + str(qualityCut) )
    axes[0, 1].grid(visible=True, which='both')
    axes[0, 1].axhline(y=0.5)
    axes[0, 1].axhline(y=0.85)
    axes[0, 1].axvline(x=ptCut)
    axes[0, 1].set_xlim([0,ptMax])
    axes[0, 1].set_ylim([0.0,1.05])
    axes[0, 1].set_xlabel(r'$p_{T}^{GEN}$')
    axes[0, 1].set_ylabel('Efficiency')
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.5)
    plt.savefig(plotDir + "/TurnOn_ptCut_{}.png".format(ptCut), bbox_inches="tight")   
    
    
    indexPlateau = ptHistoBins1.tolist().index(ptCut + 20) #np.where(ptHistoBins1 == ptCut + 20)
    effAtPlateau = np.sum(numerator_NN[indexPlateau : ]) / np.sum(denominator[indexPlateau : ])
    
    effStr = "ptCut: {} qualityCut: {}, eff@plateau: {:3.2f}".format(ptCut, qualityCut, effAtPlateau)
    return effStr, ptCut, effAtPlateau
###################################################
###################################################   
def plotPull(df, minX=-1, maxX=2, nBins=50):
       
    pull_OMTF = (df["OMTF_pt"] - df["GEN_pt"])/df["GEN_pt"]
    pull_NN = (df["NN_pt"] - df["GEN_pt"])/df["GEN_pt"]

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))  
    axes[0].hist(pull_NN, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[0].hist(pull_OMTF, range=(minX, maxX), bins = nBins, color="tomato", label="OMTF")
    axes[0].set_xlabel("(Model - True)/True")
    axes[0].legend(loc='upper right')
    axes[0].set_xlim([minX, maxX])
    
    axes[1].hist(pull_OMTF, range=(minX, maxX), bins = nBins, color="tomato", label="OMTF")
    axes[1].hist(pull_NN, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[1].set_xlabel("(Model - True)/True")
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([minX, maxX])
    plt.savefig("fig_png/Pull.png", bbox_inches="tight")
###################################################
################################################### 
def plotSingleCM(gen_labels, model_labels, modelName, palette, annot, axis):
    
    vmax = 1.0
    cm = tf.math.confusion_matrix(gen_labels, model_labels)
    cm = tf.cast(cm, dtype=tf.float32)
    cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])
    cm = tf.transpose(cm)
    
    if cm.shape[0]==2:
        vmax = 1.0
        sns.heatmap(cm, ax = axis, vmax = vmax, annot=annot, xticklabels=("-1", "1"), yticklabels=("-1", "1"), cmap=palette)
        axis.set_ylabel(r'$q^{REC}$')
        axis.set_xlabel(r'$q^{GEN}$')
    else:
        vmax = 1.0
        sns.heatmap(cm, ax = axis, vmax = vmax, annot=annot, xticklabels=5, yticklabels=5, cmap=palette)
        axis.set_ylabel(r'$p_{T}^{REC} \rm{[bin ~number]}$')
        axis.set_xlabel(r'$p_{T}^{GEN} \rm{[bin ~number]}$')
        axis.grid()
        
    axis.set_title(modelName)   
    
    max_label = np.amax([gen_labels,model_labels])+1
    axis.set_ylim([0,max_label])
    axis.set_xlim([0,max_label])
    axis.set_aspect(aspect='equal')
    axis.set_title(modelName)    
###################################################
###################################################
def plotCM(df, pT2Label):
     
    gen_labels = pT2Label(df["GEN_pt"])  
    NN_labels = pT2Label(df["NN_pt"])
    OMTF_labels = pT2Label(df["OMTF_pt"])
 
    fig, axes = plt.subplots(2, 2, figsize = (10, 10))  
    myPalette = sns.color_palette("YlGnBu", n_colors=20)
    myPalette[0] = (1,1,1)
     
    gen_labels = pT2Label(df["GEN_pt"])  
    NN_labels = pT2Label(df["NN_pt"])
    OMTF_labels = pT2Label(df["OMTF_pt"])
    
    plotSingleCM(gen_labels, NN_labels, "NN", myPalette, False, axes[0,0])
    plotSingleCM(gen_labels, OMTF_labels, "OMTF", myPalette, False, axes[0,1])
          
    gen_labels = df["GEN_charge"]  
    NN_labels = df["NN_charge"]
    OMTF_labels = df["OMTF_charge"]
    
    plotSingleCM(gen_labels, NN_labels, "NN", myPalette, True, axes[1,0])
    plotSingleCM(gen_labels, OMTF_labels, "OMTF", myPalette, True, axes[1,1])
     
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.25, hspace=0.35)
    plt.savefig("fig_png/CM.png", bbox_inches="tight")
###################################################
###################################################
def getVxMuRate(x):
    
    #Some newer parametriation do not remember source
    params = np.array([-0.235801, -2.82346, 17.162])
    integratedRate = np.power(x,params[0]*np.log(x) + params[1])*np.exp(params[2])  
    differentialRate = -np.power(x,params[0]*np.log(x) + params[1] -1)*np.exp(params[2])*(2*params[0]*np.log(x)+params[1])
    
    ##RPCConst.h parametrisation from CMS-TN-1995/150
    dabseta = 1.23 - 0.8
    lum = 1.0
    dpt = 1.0;
    afactor = 1.0e-34*lum*dabseta*dpt
    a  = 2*1.3084E6;
    mu=-0.725;
    sigma=0.4333;
    s2=2*sigma*sigma;
   
    ptlog10 = np.log10(x);
    ex = (ptlog10-mu)*(ptlog10-mu)/s2;
    rate = (a * np.exp(-ex) * afactor); 
    ######
    
    return differentialRate
###################################################
###################################################
def getVsMuRateWeight(x, hist, bins):
       
    weightToFlatSpectrum = np.divide(1.0, hist, out=np.zeros(hist.shape), where=hist>0)  
    binNumber = np.digitize(x,bins) -1  
    weight = getVxMuRate(x)*weightToFlatSpectrum[binNumber]    
    return weight
###################################################
###################################################
def plotRate(df, qualityCut, plotDir):
    
    from matplotlib.gridspec import GridSpec
        
    ptHistoBins = np.concatenate((np.arange(2,201,1), [9999]))  
    genPtHist, bin_edges = np.histogram(df["GEN_pt"], bins=ptHistoBins) 
    weights = getVsMuRateWeight(df["GEN_pt"], genPtHist, bin_edges)
       
    genPtHist_weight, bin_edges = np.histogram(df["GEN_pt"], bins=ptHistoBins, weights=weights) 
    genPtHist_weight = np.sum(genPtHist_weight) - np.cumsum(genPtHist_weight)
    
    omtfPtHist_weight, bin_edges = np.histogram(df["OMTF_pt"], bins=ptHistoBins, weights=weights) 
    omtfPtHist_weight = np.sum(omtfPtHist_weight) - np.cumsum(omtfPtHist_weight)
    
    nnPts = np.where(df["OMTF_quality"]>=qualityCut, df["NN_pt"], 0)
    nnPtHist_weight, bin_edges = np.histogram(nnPts, bins=ptHistoBins, weights=weights) 
    nnPtHist_weight = np.sum(nnPtHist_weight) - np.cumsum(nnPtHist_weight)
        
    #fig, axes = plt.subplots(2, 1, sharex=True)
    #fig.subplots_adjust(hspace=0.1)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(6, 6, figure=fig)
    axes = [0,0]
    axes[0] = fig.add_subplot(gs[0:4, : ])
    axes[1] = fig.add_subplot(gs[5: , : ])
    
    axes[0].step(ptHistoBins[:-1], genPtHist_weight, label="Vxmurate", linewidth=3, color="black", where='post')
    axes[0].step(ptHistoBins[:-1], omtfPtHist_weight, label="OMTF", linewidth=3, color="r", where='post')
    axes[0].step(ptHistoBins[:-1], nnPtHist_weight, label="NN quality >= " + str(qualityCut), linewidth=3, color="b", where='post')
    axes[0].set_xlim([2,60])
    axes[0].set_ylim([10,1E6])
    axes[0].set_ylabel('Rate [arb. units]')
    axes[0].legend(loc='upper right')
    axes[0].grid()
    axes[0].set_yscale("log")
      
    ratio = np.divide(omtfPtHist_weight, nnPtHist_weight, out=np.zeros_like(nnPtHist_weight), where=nnPtHist_weight>0)  
    axes[1].step(ptHistoBins[:-1], ratio, label="OMTF/NN", linewidth=3, color="black", where='post')
    axes[1].set_xlim([2,60])
    axes[1].set_ylim([0.9,2.1])
    axes[1].set_xlabel(r'$p_{T}^{cut}$')
    axes[1].set_ylabel('OMTF/NN')
    #axes[1].legend(loc='upper right')
    axes[1].grid()
    
    #plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.5)
    plt.savefig(plotDir + "/Rate.png", bbox_inches="tight")
    
    index10GEv = np.where(bin_edges == 10)
    index20GEv = np.where(bin_edges == 20)
    ratesStr = "Rate @10 GeV:{:.0f}, @20 GeV: {:.0f}".format(nnPtHist_weight[index10GEv][0], nnPtHist_weight[index20GEv][0])
    return ratesStr
###################################################
###################################################
def plotPtGenPtRec(df, plotDir, oneOverPt=False) :
    fig1, ax1 = plt.subplots(2, 2, figsize=(16, 9))
    
    #if oneOverPt :
    #    #ax1[0, 0].scatter(df["GEN_pt"], df["NN_pt"], linewidth=1.0, s=2)
    #    ax1[0, 0].hist2d(x = 1./df["GEN_pt"], y = 1./df["NN_pt"], bins=(50, 50), cmap=plt.cm.jet, norm="log")
    
    ptMax = 200
    ax1[1, 0].hist(x = df["GEN_pt"], weights=df["weight"], bins=1000, range=[0, ptMax], log=True, label="weighted", color='tab:red')
    ax1[1, 0].hist(x = df["GEN_pt"],                       bins=1000, range=[0, ptMax], log=True, label="org_dist")
    ax1[1, 0].set_xlabel("ptGen")
    ax1[1, 0].legend()
    
    #ax1[0, 1].scatter(1./df["GEN_pt"], 1./df["NN_pt"], linewidth=1.0, s=2)
    #ax1[0, 1].scatter(df["GEN_pt"], df["NN_pt"], linewidth=1.0, s=2)
    ax1[0, 1].hist2d(x = df["GEN_pt"], y = df["NN_pt"], weights=df["weight"], 
                     bins=(50, 50), range=[[0, ptMax], [0, 200]], cmap=plt.cm.jet, norm=colors.LogNorm())
    
    ax1[1, 1].hist2d(x = df["GEN_pt"], y = df["OMTF_pt"], weights=df["weight"], 
                     bins=(50, 50), range=[[0, ptMax], [0, 200]], cmap=plt.cm.jet, norm=colors.LogNorm())
    
    binWidth = 2
    bins=np.array(range(0,202, binWidth))
    mean_y_pulls, mean_y_abs_pulls, mean_y_squared_pulls = getPullsInBins(df["GEN_pt"], df["NN_pt"], bins)
    
    mean_y_pulls = mean_y_pulls / (bins + binWidth/2)
    mean_y_abs_pulls = mean_y_abs_pulls / (bins + binWidth/2)
    mean_y_squared_pulls = mean_y_squared_pulls / (bins + binWidth/2)
    
    ax1[0, 0].plot(bins, mean_y_pulls, "k.", label="NN_pt - genPt")#, color="black")
    ax1[0, 0].plot(bins, mean_y_abs_pulls, "g.", label="|NN_pt - genPt|")#, color="red")
    ax1[0, 0].plot(bins, mean_y_squared_pulls, "r.", label="(NN_pt - genPt)^2")#, color="blue")
    ax1[0, 0].grid(visible=True, which='both')
    #ax1[0, 0].set_ylim([-350, 500])
    ax1[0, 0].set_ylim([-2.5, 3.5])
    ax1[0, 0].legend()
    
    plt.savefig(plotDir + "/genPt_NN_pt.png", bbox_inches="tight")
###################################################
###################################################
def ptRecalibration(df, plotDir, oneOverPt, label) :
    ptMax = 200.
    binCnt = 800 
    ptBinWidth = ptMax / binCnt #[GeV]
    if oneOverPt :
        y = 1./df["GEN_pt"]
        x = 1./df[label]
    else :
        y = df["GEN_pt"]
        x = df[label] 
    
    ptGenPtRec, xedges, yedges = np.histogram2d(x = x, y = y, bins=(binCnt, binCnt), range=[[0, ptMax], [0, ptMax]])
    print("ptGenPtRec.shape", ptGenPtRec.shape)
    print("xedges.shape", xedges.shape)
    print("yedges.shape", yedges.shape)
    
    
    effTreshold = 0.8
    plateauOffset = int(20. / ptBinWidth)
    
    ptToPtCalib = np.zeros(binCnt + 1)
    #for iPt1 in range(0, binCnt - plateauOffset -1) :
    for iPt1 in reversed(range(0, binCnt - plateauOffset-1)) : #iPt1 is ptCut here
        accpeptedEvCntOnPlateau = np.sum(ptGenPtRec[iPt1 : , iPt1 + plateauOffset : ]) # [x, y] i.e. [ptRec, ptGen]
        allEvCntOnPlateau       = np.sum(ptGenPtRec[     : , iPt1 + plateauOffset : ]) # [x, y] i.e. [ptRec, ptGen]
        
        if allEvCntOnPlateau <= 0 :
            continue
        
        plateauEff = accpeptedEvCntOnPlateau / allEvCntOnPlateau
        
        #print("iPt1", iPt1, "pt1", iPt1 * ptBinWidth, "accpeptedEvCntOnPlateau", accpeptedEvCntOnPlateau, "allEvCntOnPlateau", allEvCntOnPlateau, "plateauEff", plateauEff)
        #print("iPt1", iPt1, "pt1", iPt1 * ptBinWidth, "plateauEff", plateauEff)
        
        previousBinEff = 0
        for iPt2 in range(0, binCnt -1) : 
            #iPt2 indexes the ptGen here
            accpeptedEvCntInBin = np.sum(ptGenPtRec[iPt1 : , iPt2])  #[y, x] i.e. [ptRec, ptGen]
            allEvCntInBin       = np.sum(ptGenPtRec[     : , iPt2])
            
            if allEvCntInBin == 0 :
                continue
            
            binEff = accpeptedEvCntInBin / allEvCntInBin
            
            #if accpeptedEvCntInBin > 0 or allEvCntInBin > 0:
            #    print("iPt2", iPt2, "iPt1", iPt1, "accpeptedEvCntInBin", accpeptedEvCntInBin, "allEvCntInBin", allEvCntInBin, "binEff", binEff)
         
            if (binEff > (effTreshold * plateauEff) ) :  
                #prevBinContent = ptToPtCalib->GetBinContent(iPt1, ptGenVsPtLutNN->GetXaxis()->GetBinLowEdge(iPt2));
                #if(prevBinContent != 0) 
                #    std::cout<<"PtCalibration::train: iPt1 "<<iPt1<<"iPt2 "<<iPt2<<" prevBinContent "<<prevBinContent<<" !!!!!!!!!!!!!!!!!!";

                if( (binEff - (effTreshold * plateauEff) ) < ( (effTreshold * plateauEff) - previousBinEff) ) :
                    #ptToPtCalib->SetBinContent(iPt1, ptGenVsPtLutNN->GetXaxis()->GetBinLowEdge(iPt2))
                    ptToPtCalib[iPt1] = xedges[iPt2]
                    #print("iPt2", iPt2, "pt2", iPt2 * ptBinWidth, "binEff", binEff, "xedges[iPt2]", xedges[iPt2], "\n")
                else :
                    #ptToPtCalib->SetBinContent(iPt1, ptGenVsPtLutNN->GetXaxis()->GetBinLowEdge(iPt2-1))
                    ptToPtCalib[iPt1] = xedges[iPt2-1]
                    #print("iPt2", iPt2, "pt2", iPt2 * ptBinWidth, "binEff", binEff, "xedges[iPt2-1]", xedges[iPt2-1], "\n")
                    
                break

            previousBinEff = binEff;    
        
            
    for iPt in range( int(100./ptBinWidth), binCnt+1) :
        ptToPtCalib[iPt] = iPt * ptBinWidth + (ptToPtCalib[int(100./ptBinWidth)] - 100)       
            
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xedges, ptToPtCalib, linewidth=1.0)
    ax.set_xlim(0, ptMax * 1.1)
    ax.set_ylim(0, ptMax * 1.1)
    
    plt.savefig(plotDir + "/ptToPtCalib_" + label + ".png", bbox_inches="tight")
    
    #print("ptToPtCalib shape", ptToPtCalib.shape, "\n", ptToPtCalib)
    #print("xedges shape", xedges.shape, "\n", xedges)
    
    return ptToPtCalib, xedges

def ptRecalibrated(pt, ptToPtCalib, xedges) :
    pt = np.where(pt < 199, pt, 199)
    return ptToPtCalib[np.digitize(pt, xedges)]
    
    
def getPullsInBins(x, y, bins):
    # Use numpy's digitize function to assign each x value to a bin
    bin_indices = np.digitize(x, bins)
    # Initialize an array to store the mean y value for each bin

    # Initialize an array to store the number of elements in each bin    
    mean_y_pulls = np.zeros(len(bins))
    mean_y_abs_pulls = np.zeros(len(bins))
    mean_y_squared_pulls = np.zeros(len(bins))
    bin_counts = np.zeros(len(bins))

    # Iterate through all x, y values and add y values to the appropriate bin
    for i in range(len(x)):
        #todo: if needed, add a protection vs bin_indices[i] = 0, which corrsponds to the x < bins[0]
        mean_y_pulls[bin_indices[i]-1] += (y[i] - x[i])
        mean_y_abs_pulls[bin_indices[i]-1] += abs(y[i] - x[i])
        mean_y_squared_pulls[bin_indices[i]-1] += (y[i] - x[i])**2
        
        bin_counts[bin_indices[i]-1] += 1
        
        #if abs(y[i] - x[i]) > 100 :
        #   print("i", i, "ptGeN", x[i], "ptNN", y[i]) 
        
    # Divide the sum of y values in each bin by the number of elements in the bin to get the mean
    for i in range(len(bins)):
        if bin_counts[i] > 0:
            mean_y_pulls[i] = mean_y_pulls[i] / bin_counts[i]
            mean_y_abs_pulls[i] = mean_y_abs_pulls[i] / bin_counts[i]
            mean_y_squared_pulls[i] = np.sqrt(mean_y_squared_pulls[i] / bin_counts[i] )
    
    #mean_y_pulls = np.divide(mean_y_pulls[i], bin_counts[i], where = (bin_counts[i] > 0))
    
    return mean_y_pulls, mean_y_abs_pulls, mean_y_squared_pulls    
    