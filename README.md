# Gaussian Mixture Model (GMM) clustering analysis based on the measured mScarlett and nuclear TDP-43 signal of TDP-REG SH-SY5Y cells post TDP-43 knockdown via shRNA

Here we describe the process of generating the clustering analysis from cells with TDP-43 knockdown and the activation of TDP-REG reporter as in our manuscript (Fig.S5F)
To use, orient to the folder that contains the script, then 
`python GMM_on_LV_TDP_KD.py rep1.xlsx`
or with any input data file that you would like to process.

## Data
The raw data file was outputed from Imaris 10.2. After segmenting the nuclear surface in the images, the intensity of mScarlet and TDP-43 staining was extracted, together with the volume of the nuclear space of each cell. Method of staining, image aquisition, deconvolution and parameters for segmentations are decribed in the Material & method part of out publication.
The raw Imaris output need to be preprocessed in order to have for each replication of the experiment, all objects (nuclei) from all images combined, with each row represent every object and the log transformed nuclear TDP intensity (column name "nucTDP_intensity_normalized_log" in the first column, as well as the log transformed nuclear mScarlet intensity (column name "reporter_intensity_normalized_log") in the second column.
You can use the R script [Preprocessing_Imaris_output.Rmd](./Preprocessing_Imaris_output.Rmd) to reproduce the results.

## GMM modeling
After examining all the data points (Figure 1 in the output), we decided to remove the outliers in the data which has extreme low intensities for mScarlet or TDP, as those are essentially dead cells or segmentation artifacts. 
For testing the optimal number of components to fit in the data, we ploted the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC) (Figure 2) and decided to use 2 components for the goodness of fit and to avoid over fitting.
By fitting with the *GaussianMixture* class from *scikit-learn* we were able to extract the mean of the two clusters, as well as the boundery where one could unambigously identify the cluster of cells with activation.
You can use the python script [GMM_on_LV_TDP_KD.py](./GMM_on_LV_TDP_KD.py) to reproduce the results.



