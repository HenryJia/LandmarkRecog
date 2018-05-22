# Kaggle Landmark Recognition Challenge in PyTorch

## Method
We use an attention based model similar to DELF, but we do not use PCA to reduce dimensionality. The model uses pretrained ResNet50 weights on ImageNet


## Files:
* clean_df.py
  * Script to clean the dataframe by checking every file is there and openable, deleting all the entries that do not exist
* data_utils.py
  * Contains various functions and classes for loading data among other stuff
* download_all.py
  * Script to download all of the data
* models.py
  * Contains the PyTorch code for our model
* split_data.py
  * Script to split the trainset into a train and validation set, keeping class ratios as balanced as we can
* submit-NN.py
  * Script to generate the submission
* train-NN.py
  * Script to train the model
* train_utils.py
  * Contains various useful utility functions for training

## References

```H. Noh, A. Araujo, J. Sim, T. Weyand, B. Han, "Large-Scale Image Retrieval with Attentive Deep Local Features", Proc. ICCV'17```
