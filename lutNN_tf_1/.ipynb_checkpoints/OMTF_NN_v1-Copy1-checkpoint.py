else :
    if lut_nn :
        num_inputs = networkInputSize
        
        model = keras.Sequential()
        layer1 = LutInterLayer.LutInterLayer("layer1", lut_size = layer1_lut_size, num_inputs = num_inputs, num_outputs = layer1_neurons, input_offset= 0, write_lut_hist=True, last_input_is_bias = last_input_is_bias)
        layer2 = LutInterLayer.LutInterLayer("layer2", lut_size = layer2_lut_size, num_inputs = layer1_neurons, num_outputs = layer2_neurons, input_offset = layer2_input_offset, write_lut_hist=True)
        #layer3 = LutInterLayer.LutInterLayer("layer3", lut_size = layer3_lut_size, num_inputs = layer2_neurons, num_outputs = 1)
        layer3 = LutInterLayer.LutInterLayer("layer3", lut_size = layer3_lut_size, num_inputs = layer2_neurons, num_outputs = 1, write_lut_hist=True)

        print("building model")
        model.add(tf.keras.Input(shape = [num_inputs], name="delta_Phi"))
        model.add(layer1)
        model.add(layer2)
        model.add(layer3)
        
        #job_dir = "training/2022_12_20-11_09_57"
        #job_dir = "training/2022_12_20-00_36_27"
        #job_dir = "training/2022_12_29-08_25_50_lut_nn_16_8_1"
        #job_dir = "training/2022_12_29-10_33_17_lut_nn_32_16_1"
        #job_dir = "training/2022_12_31-14_36_14_lut_nn_16_8_1"
        #job_dir = "training/2022_12_29-10_33_17_lut_nn_32_16_1"
        #job_dir = "training/2022_12_29-08_25_50_lut_nn_16_8_1"
        job_dir = "training/2023_01_02-00_38_32_lut_nn_16_8_1"
        #job_dir = "training/2023_01_02-15_54_50_lut_nn_16_8_1"
        
        #job_dir = "training/2023_01_07-09_14_28_lut_nn_32_16_1//"
        #job_dir = "training/2023_01_07-19_04_30_lut_nn_32_16_1/"
        #job_dir = "training/2023_01_07-21_20_43_lut_nn_32_16_1/"
        
        #job_dir = "training/2023_01_08-01_03_33_lut_nn_16_8_1/"
                
        filepath = job_dir + "/cp-0012.ckpt"
        model.load_weights(filepath)
        
        model.compile(loss = loss_fn)
    else:    
        #job_dir = "training/2022_12_19-21_56_22"
        
        #job_dir = "training/2023_01_05-23_04_09_classic_192_96_1"
        #model = "0012_2023_Jan_06-00_46" 
        
        #job_dir = "training/2023_01_03-08_18_00_classic_256_128_1"
        #model = "0012_2023_Jan_03-10_03"       
        
        #job_dir = "training/2023_01_03-14_56_23_classic_256_128_1"
        #model = "0012_2023_Jan_03-16_41"
        
        #job_dir = "training/2023_01_06-13_14_24_classic_256_128_128_1"
        #model = "0012_2023_Jan_06-15_04"
        
        #job_dir = "training/2023_01_10-14_58_48_classic_256_128_128_1"
        #model = "0015_2023_Jan_10-17_19"

        job_dir = "training/2023_01_30-13_34_03_classic_256_128_96_3"
        #model = "0005_2023_Jan_30-13_14"

        #filepath = job_dir + "/" + model
        #model = tf.keras.models.load_model(filepath)
        
        model = keras.Sequential()
        num_inputs = 2 * np.sum(getFeaturesMask()) + 1
        model.add(tf.keras.Input(shape = [num_inputs], name="delta_Phi") )
        model.add(tf.keras.layers.Dense(dense_layer1_size, activation='relu', name="pt_layer_1") )
        model.add(tf.keras.layers.Dense(dense_layer2_size, activation='relu', name="pt_layer_2") )
        model.add(tf.keras.layers.Dense(dense_layer3_size, activation='relu', name="pt_layer_3") )
        model.add(tf.keras.layers.Dense(dense_layer4_size)) #, activation='sigmoid'
        
        filepath = job_dir + "/cp-0005.ckpt"
        model.load_weights(filepath)
        
        model.compile(loss = loss_fn)
        
    #model.load_weights(filepath, by_name)
    print("model loaded from", job_dir)
    model.summary()
    
for layer in model.layers:
    print(layer)
    if isinstance(layer, LutInterLayer.LutInterLayer):
        print("layer.write_lut_hist", layer.write_lut_hist)
        layer.write_lut_hist = True
        print("layer.write_lut_hist", layer.write_lut_hist)
    

if False :
    test_dataset = loadDataset(testFileNames, isTrain=True, nEpochs=1, batchSize= 32*1024) #32*

    model.evaluate(test_dataset, batch_size = 1024 * 2,
              #callbacks=[tensorboard_callback, cp_callback, LossToTBoardCallback() ]
              )
    
    exit()

test_dataset = loadDataset(testFileNames, isTrain=False, nEpochs=1, batchSize= 32*1024) #32*
    
df = pd.DataFrame(columns=["genPt", "genCharge", "genEta",
                            "OMTF_pt", "OMTF_charge",
                            "NN_pt" #,"NN_charge","NN_prob"
                            ])

print("test_dataset", test_dataset)

count = 0
for aBatch in test_dataset.as_numpy_iterator(): #TODO change to 
    df = fillPandasDatasetRegression(aBatch, df)
    count = count +1
    #if count >= 10 :
     #   break


plotDir = job_dir + "/figures"
if not train :
    plotDir = job_dir + "/figures_1"
    if not os.path.exists(plotDir):
        os.mkdir(plotDir, )
else :    
    os.mkdir(plotDir)

#doing te plot_luts takes long time, so it is not worth to reapeat it
#remove if if needed     
if train :    
    for layer in model.layers:
        if isinstance(layer, LutInterLayer.LutInterLayer):
            layer.plot_luts(plotDir + "/")

print("df.size", df.size)

print("df[genPt].size", df["genPt"].size)
print("genPt\n", df["genPt"])
print("NN_pt\n", df["NN_pt"])

plt.style.use('_mpl-gallery')
plt.rcParams['axes.labelsize'] = 6 
plt.rcParams['ytick.labelsize'] = 6 
plt.rcParams['xtick.labelsize'] = 6 
# plot



if oneOverPt :
    #if in modifyFeatures ptLabels =  tf.divide(1., ptLabels)
    df["genPt"] = 1./df["genPt"]
    df["NN_pt"] =  1./df["NN_pt"]


losses_text_file = open(job_dir + "/losses.txt", "w")

lossFunction = tf.keras.losses.MeanAbsoluteError()
loss = lossFunction(df["genPt"], df["NN_pt"]).numpy()
print("MeanAbsoluteError NN", loss)
print("MeanAbsoluteError NN", loss, file=losses_text_file)

loss = lossFunction(df["genPt"], df["OMTF_pt"]).numpy()
print("MeanAbsoluteError OMTF", loss)

lossFunction = tf.keras.losses.MeanSquaredError()
loss = lossFunction(df["genPt"], df["NN_pt"]).numpy()
print("MeanSquaredError NN", loss)
print("MeanSquaredError NN", loss, file=losses_text_file)

loss = lossFunction(df["genPt"], df["OMTF_pt"]).numpy()
print("MeanSquaredError OMTF", loss)


plf.plotPtGenPtRec(df, plotDir, oneOverPt)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! pt recalibration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ptToPtCalibNN, xedges = plf.ptRecalibration(df, plotDir, oneOverPt, "NN_pt")
df["NN_pt"] = plf.ptRecalibrated(df["NN_pt"], ptToPtCalibNN, xedges) 


lossFunction = tf.keras.losses.MeanAbsoluteError()
loss = lossFunction(df["genPt"], df["NN_pt"]).numpy()
print("ptRecalibrated\nMeanAbsoluteError NN", loss)
print("ptRecalibrated\nMeanAbsoluteError NN", loss, file=losses_text_file)

lossFunction = tf.keras.losses.MeanSquaredError()
loss = lossFunction(df["genPt"], df["NN_pt"]).numpy()
print("MeanSquaredError NN", loss)
print("MeanSquaredError NN", loss, file=losses_text_file)


#ptToPtCalibNN, xedges = plf.ptRecalibration(df, plotDir, oneOverPt, "OMTF_pt")
#df["OMTF_pt"] = plf.ptRecalibrated(df["OMTF_pt"], ptToPtCalibNN, xedges)

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

