#' Feature based model averaging with xgboost.
#' @param ts.data targeted time series to be predicted(testing data,such as M4).
#' @param files.of.features.of.train.data files of features of training time series and features are saved in csv.Data format can be seen in the file "input.dataformat".
#' @param dimension.of.features dimension of time series features
#' @param files.of.owa.of.train.data files of owa(prediction error) of training time series and owa are saved in csv.Data format can be seen in the file "input.dataformat".
#' @param files.of.features.of.test.data files of features of testing time series and features are saved in csv.Data format can be seen in the file "input.dataformat".
#' @param files.of.prediction.value.of.different.methods files of prediction values of testing time series and prediction values are saved in csv.Data format can be seen in the file "input.dataformat".
#' @return a dataframe of owa for the test time series datasets
#' 
#' @author Xixi Li, Yanfei Kang and Feng Li
#' @examples
#' ts.data<-M4
#' files.of.features.of.train.data<-'E:/lixixi/M4_9_methods/process/5dimension_reduction_and_model_selection/features'
#' dimension.of.features<-4200
#' files.of.owa.of.train.data<-"./m4_9_methods_1-10000/train/1-100000/train_errors_new"
#' files.of.features.of.test.data<-'E:/lixixi/M4_9_methods/process/5dimension_reduction_and_model_selection/test_features'
#' files.of.prediction.value.of.different.methods<-"./m4_9_methods_1-10000/test/1-100000/prediction.value"
#' owa<-features.and.owa.of.train.data(ts.data,files.of.features.of.train.data,dimension.of.features,files.of.owa.of.train.data,
#'                                   files.of.features.of.test.data,files.of.prediction.value.of.different.methods)
#' @export
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('subset_methods.R')
source('process_dataset.R')
source('hyperparam.R')
source('generate_classif_problem.R')
source('forec_methods_list.R')
source('ensemble_classifier.R')
source('combination_ensemble.R')
source('forec_methods_list.R')
library(M4comp2018)
library(forecast)
library(parallel)

detectCores()   # 4 core
set.seed(6633-2018)
library(Tcomp)

#get some basic data: features of training and testing data, owa of training data and prediction value of 9 method of testing data
#get some basic data: features of training and testing data, owa of training data and prediction value of 9 method of testing data
get.features.for.batch.ts<-function(ts.dataset,file.path.of.features){
  library(tidyverse)
  #features shape: the first column is id
  features.df=read.csv(file.path.of.features,header = TRUE,stringsAsFactors = FALSE)
  for (i in 1:length(ts.dataset)) {
    #ts.dataset[[i]]$features<-features.df[which(features.df$id == ts.dataset[[i]]$st), ][1,-1]
    #id.num=ts.dataset[[i]]$st
    #f<-features.df%>%filter(id==id.num)
    #ts.dataset[[i]]$features<-f[1,-1]
    ts.dataset[[i]]$features<-features.df[i,-1]
    
    #print(features.df[which(features.df$id == ts.dataset[[i]]$st), ][1,-1][1,100])
  }
  return(ts.dataset)
}


#data.type.of.training.data:'Monthly'
#data.type.of.testing.data:'MONTHLY'
#training.dataset:M4
#testing.dataset:tourism
#file.path.of.training.data.features
#file.path.of.training.data.features='/Users/xushengxiang/Desktop/lixixi/M4_model_selection_averaging/ts-image-forecasting/code/data/Tourism/feature/tourism-train-feature.csv'
#file.path.of.testing.data.features
#file.path.of.testing.data.features='/Users/xushengxiang/Desktop/lixixi/M4_model_selection_averaging/ts-image-forecasting/code/data/Tourism/feature/tourism-train-feature.csv'
image.based.model.combination<-function(data.type.of.training.data,
                                        data.type.of.testing.data,
                                        training.dataset,
                                        testing.dataset,
                                        feature.type,
                                        file.path.of.training.data.features,
                                        file.path.of.testing.data.features,
                                        file.path.of.training.data.prediction.value,
                                        file.path.of.testing.data.prediction.value){
  #extract monthly series as training data
  print('load prediction value of training data')
  training_data <- Filter(function(l) l$period == data.type.of.training.data, training.dataset)
  #training_data <- calc_forecasts(training_data, forec_methods(), n.cores=3)
  #load all prediction value of M4 training data
  # forecasting.value=paste(data.type,'ff',sep = '_')
  # forecasting.value.rda=paste(forecasting.value,'rda',sep = '.')
  #'Monthly_ff.rda'
  load(file.path.of.training.data.prediction.value)
  # ff.list[[100]]$st
  # length(ff.list)
  for (i in 1:length(ff.list)) {
    training_data[[i]]$ff=ff.list[[i]]$ff
  }
  #get monthly 
  #tourism <- calc_forecasts(tourism, forec_methods(), n.cores=3)
  #get prediction value from local rda file
  #'tourism.rda'
  load(file.path.of.testing.data.prediction.value)
  print('load prediction value of testing data')
  testing_data<- Filter(function(l) l$period == data.type.of.testing.data, testing.dataset)
  for (i in 1:length(ff.list)) {
    testing_data[[i]]$ff=ff.list[[i]]$ff
  }
  
  #tourism <- calc_forecasts(tourism, forec_methods(), n.cores=3)
  
  #2.get features of training and testing data
  #2.1 for M4 monthly data
  print('load features of training data')
  training_data=get.features.for.batch.ts(training_data,file.path.of.training.data.features)
  #2.2 for tourism monlty data
  print('load features of testing data')
  testing_data=get.features.for.batch.ts(testing_data,file.path.of.testing.data.features)
  test=testing_data[[200]]$features
  
  #3.calculte errors of training data
  training_data=calc_errors_new(training_data)
  #4.begin to contrct feature and owa for xgboost
  train_data <- create_feat_classif_problem(training_data)
  ggg=round(head(train_data$data, n=3),2)
  #5 begin to train the xgboost with custom loss function
  print('training model..')
  meta_model <- train_selection_ensemble(train_data$data, train_data$errors)
  #6.testing the model
  test_data <- create_feat_classif_problem(testing_data)
  preds <- predict_selection_ensemble(meta_model, test_data$data)
  head(preds)
  #7.model combination
  print('model combination')
  testing_data <- ensemble_forecast(preds, testing_data)
  testing_data[[1]]$y_hat
  #8.compute the performance
  mape_all_new<-c()
  mase_all_new<-c()
  for(r in 1:length(testing_data)){
    #print(r)
    mape_h<-mape_cal(testing_data[[r]]$xx,testing_data[[r]]$y_hat)
    mape_temp<-mean(mape_h)
    mape_all_new<-c(mape_all_new,mape_temp)
    
    mase_h<-mase_cal(testing_data[[r]]$x,testing_data[[r]]$xx,testing_data[[r]]$y_hat)
    #mase_temp<-tail(mase_h,1)
    mase_temp<-mean(mase_h)
    mase_all_new<-c(mase_all_new,mase_temp)
  }
  print(feature.type)
  print('mape')
  print(mean(mape_all_new))
  print('mase')
  print(mean(mase_all_new))
  forecasts.list=list()
  #data.temp <- Filter(function(l) l$period == data.type[i], tourism)
  for (j in 1:length(testing_data)) {
    forecasts.list[[j]]=list(st=testing_data[[j]]$st,y_hat=testing_data[[j]]$y_hat)
  }
  #save to rda file
  file.temp=paste(data.type.of.testing.data,feature.type,sep = '_')
  file1= paste(file.temp,'_y_hat.rda',sep = '')
  file2=paste('intermediate.results/',file1,sep = '')
  save(forecasts.list, file = file2)
  
}

# #1.inception V1
# #test the code
# data.type.of.training.data='Monthly'
# data.type.of.testing.data='MONTHLY'
# training.dataset=M4
# testing.dataset=tourism
# feature.type='inception_v1'
# file.path.of.training.data.features='C:/xixi/feature_extraction/cnn/cnn-features/M4/Monthly-train-feature-inception_v1.csv'
# file.path.of.testing.data.features='C:/xixi/feature_extraction/cnn/cnn-features/Tourism/tourism-monthly-train-feature-inception_v1.csv'
# file.path.of.training.data.prediction.value='./forecasts/M4/Monthly_ff.rda'
# file.path.of.testing.data.prediction.value='./forecasts/Tourism/MONTHLY_ff.rda'
# image.based.model.combination(data.type.of.training.data,
#                               data.type.of.testing.data,
#                               training.dataset,
#                               testing.dataset,
#                               feature.type,
#                               file.path.of.training.data.features,
#                               file.path.of.testing.data.features,
#                               file.path.of.training.data.prediction.value,
#                               file.path.of.testing.data.prediction.value)

#2.resnet50
#test the code
data.type.of.training.data='Monthly'
data.type.of.testing.data='MONTHLY'
training.dataset=M4
testing.dataset=tourism
feature.type='resnet50'
file.path.of.training.data.features='C:/xixi/feature_extraction/cnn/cnn-features/M4/Monthly-train-feature-resnet_v1_50.csv'
file.path.of.testing.data.features='C:/xixi/feature_extraction/cnn/cnn-features/Tourism/tourism-monthly-train-feature-resnet_v1_50.csv'
file.path.of.training.data.prediction.value='./forecasts/M4/Monthly_ff.rda'
file.path.of.testing.data.prediction.value='./forecasts/Tourism/MONTHLY_ff.rda'
image.based.model.combination(data.type.of.training.data,
                              data.type.of.testing.data,
                              training.dataset,
                              testing.dataset,
                              feature.type,
                              file.path.of.training.data.features,
                              file.path.of.testing.data.features,
                              file.path.of.training.data.prediction.value,
                              file.path.of.testing.data.prediction.value)

#3.resnet101
#test the code
data.type.of.training.data='Monthly'
data.type.of.testing.data='MONTHLY'
training.dataset=M4
testing.dataset=tourism
feature.type='resnet101'
file.path.of.training.data.features='C:/xixi/feature_extraction/cnn/cnn-features/M4/Monthly-train-feature-resnet_v1_101.csv'
file.path.of.testing.data.features='C:/xixi/feature_extraction/cnn/cnn-features/Tourism/tourism-monthly-train-feature-resnet_v1_101.csv'
file.path.of.training.data.prediction.value='./forecasts/M4/Monthly_ff.rda'
file.path.of.testing.data.prediction.value='./forecasts/Tourism/MONTHLY_ff.rda'
image.based.model.combination(data.type.of.training.data,
                              data.type.of.testing.data,
                              training.dataset,
                              testing.dataset,
                              feature.type,
                              file.path.of.training.data.features,
                              file.path.of.testing.data.features,
                              file.path.of.training.data.prediction.value,
                              file.path.of.testing.data.prediction.value)

#4.vgg19
#test the code
data.type.of.training.data='Monthly'
data.type.of.testing.data='MONTHLY'
training.dataset=M4
testing.dataset=tourism
feature.type='vgg19'
file.path.of.training.data.features='C:/xixi/feature_extraction/cnn/cnn-features/M4/Monthly-train-feature-vgg_19.csv'
file.path.of.testing.data.features='C:/xixi/feature_extraction/cnn/cnn-features/Tourism/tourism-monthly-train-feature-vgg_19.csv'
file.path.of.training.data.prediction.value='./forecasts/M4/Monthly_ff.rda'
file.path.of.testing.data.prediction.value='./forecasts/Tourism/MONTHLY_ff.rda'
image.based.model.combination(data.type.of.training.data,
                              data.type.of.testing.data,
                              training.dataset,
                              testing.dataset,
                              feature.type,
                              file.path.of.training.data.features,
                              file.path.of.testing.data.features,
                              file.path.of.training.data.prediction.value,
                              file.path.of.testing.data.prediction.value)

