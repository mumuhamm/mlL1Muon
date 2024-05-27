if oneOverPt :
    #if in modifyFeatures ptLabels =  tf.divide(1., ptLabels)
    df["genPt"] = 1./df["genPt"]
    df["NN_pt"] =  1./df["NN_pt"]

df1 = df

ptCuts = (10, 15, 20, 25, 30, 40)
qualityCut =  12

for ptCut in ptCuts :
    eff = plf.plotTurnOn(df1, ptCut=ptCut, qualityCut = qualityCut, plotDir=plotDir)
    print( eff[0])
    print( eff[0], file=losses_text_file)

rates = plf.plotRate(df1, qualityCut, plotDir=plotDir)

plt.show()
plt.subplots_adjust(hspace=0.4, wspace=0.17, left = 0.035, bottom = 0.045, top = 0.95)

print("rates NN", rates)
print("rates NN", rates, file=losses_text_file)
losses_text_file.close()

if train :
    shutil.copy2('OMTF_NN_v1.py', job_dir)
    shutil.copy2('out.txt', job_dir)

