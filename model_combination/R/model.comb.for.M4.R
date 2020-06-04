#train features
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(stringr)
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
#params for xgboost
params=c(20,0.66,0.890,0.870)
#path of training feature of M4 temporal holdout dataset
train.feture.file='D:/M4/train_feature/resnet101'
#path of feature of M4 dataset
file.of.testing.features='D:/M4/test_feature/resnet101'
#owa of M4 temporal holdout dataset
file.train.owa='D:/M4/train_errors_new_M4/train_errors_new_M4'
#forecasts of 9 methods for M4
file.of.testing.prediction.values="D:/M4/prediction_value_new_M4/prediction_value_new_M4"

#4_train <- M4[c(1,2,3,4)]
M4_test<-M4
M4_train <- temp_holdout(M4)
#####################train features#################
train.data.df =data.frame()
train.data.df = train.data.df[-1,]
#fileName<-dir('E:/lixixi/M4_9_methods/M4_meta_learning/sample_train_10000/')

train.file.name<-dir(train.feture.file)
count_list<-c()
file_list<-c()
data_list<-list()
i=1
for (file_name in train.file.name) {
  data_temp<-read.csv(paste(train.feture.file,sep='/',file_name))
  data_temp<-na.omit(data_temp)
  count_list<-c(count_list,length(data_temp[,1]))
  train.data.df<-rbind(train.data.df,data_temp)
  file_list<-c(file_list,file_name)
  #data_list[[i]]<-data_temp
  i=i+1
}
#na.index1=data_df[data_df!='NA']
train_features<-na.omit(train.data.df)
#make mapping from id(Y100) to index(number)
train_index_list_vec=levels(train_features$id)
train_index_list_len<-length(train_index_list_vec)
train_index_of_ts_temp=c()
for (i in 1:length(train_features$id)) {
  #get true index of one time series
  index_char_vec=str_extract_all(train_index_list_vec[i],"[0-9]")[[1]]
  index_str=str_c(index_char_vec)
  str_link=paste(index_str[1:length(index_str)],collapse="")
  j<-as.numeric(str_link)
  period_type<-strsplit(train_index_list_vec[i],"")[[1]][1]
  if(period_type=='Y'){
    index_num<-j
  }else if(period_type=='Q'){
    index_num<-j+23000
  }else if(period_type=='M'){
    index_num<-j+47000
  }else if(period_type=='W'){
    index_num<-j+95000
  }else if(period_type=='D'){
    index_num<-j+95359
  }else if(period_type=='H'){
    index_num<-j+99586
  }
  train_index_of_ts_temp=c(train_index_of_ts_temp,index_num)
  M4_train[[index_num]]$features<-train_features[i,-1]
}
ff=M4_train[[1]]$features
#features of M4_train
# for(i in 1:index_list_len){
#   print(i)
#   #add features accroding to index_list(id)
#   M4_train[[index_list_vec[i]]]$features<-train_features[i,1:d]
# }
test1=M4_train[[61759]]$features
##########################train mase##################################
#train_mase<-read.csv("E:/lixixi/M4_9_methods/process/mase_of_9_methods_using_rob/mase-9-methods-new/train-1-100000-mase-new.csv")
#index_mase<-train_mase$index_list
#index_mase_len<-length(index_mase)
###########################train errors#########################
#file.train.owa='D:/M4/train_errors/train_errors_new_M4/train_errors_new_M4'
file.of.owa<-dir(file.train.owa)
index.of.train.data=c()
for (i in 1:length(file.of.owa)) {
  #read csv mase
  forecast_of_ts<-read.csv(paste(file.train.owa,file.of.owa[i],sep = "/"))
  #get true index of one time series
  #value_file_name<-file_forecasting[e]
  index_char_vec=str_extract_all(file.of.owa[i],"[0-9]")[[1]]
  index_str=str_c(index_char_vec)
  str_link=paste(index_str[1:length(index_str)],collapse="")
  j<-as.numeric(str_link)
  period_type<-strsplit(file.of.owa[i],"")[[1]][1]
  if(period_type=='Y'){
    index_num<-j
  }else if(period_type=='Q'){
    index_num<-j+23000
  }else if(period_type=='M'){
    index_num<-j+47000
  }else if(period_type=='W'){
    index_num<-j+95000
  }else if(period_type=='D'){
    index_num<-j+95359
  }else if(period_type=='H'){
    index_num<-j+99586
  }
  index.of.train.data=c(index.of.train.data,index_num)
  #index_of_ts_temp=c(index_of_ts_temp,index_num)
  M4_train[[index_num]]$errors<-forecast_of_ts[,3]
}

M4_train_copy<-M4_train
##########delect time series who has no features
# ??x?в?ͬ??y??Ԫ??
#jiaoji<-intersect(index_temp,index_of_ts_temp)
#
index.of.M4<-seq(from=1, to=100000,by=1)
delect_index<-setdiff(index.of.M4,index.of.train.data)
#
M4_train<-M4_train[-delect_index]
M4_train[[1]]$mase_err
length(M4_train)
M4_train[[1]]$errors
#create training data
train_data <- create_feat_classif_problem(M4_train)
round(head(train_data$data, n=3),2)
round(head(train_data$errors, n=3),2)
round(head(train_data$labels, n=3),2)
set.seed(1345) #set the seed because xgboost is random!
meta_model <- train_selection_ensemble(train_data$data, train_data$errors,params)

#######################test features####################
M4_test<-M4
test_data_df =data.frame()
test_data_df = test_data_df[-1,]
#file.of.testing.features='E:/lixixi/M4_9_methods/process/5dimension_reduction_and_model_selection/test_features'
test.file.name<-dir(file.of.testing.features)
count_list<-c()
file_list<-c()
data_list<-list()
i=1
for (file_name in test.file.name) {
  data_temp<-read.csv(paste(file.of.testing.features,sep='/',file_name))
  data_temp<-na.omit(data_temp)
  count_list<-c(count_list,length(data_temp[,1]))
  test_data_df<-rbind(test_data_df,data_temp)
  file_list<-c(file_list,file_name)
  #data_list[[i]]<-data_temp
  i=i+1
}
#na.index1=data_df[data_df!='NA']
test_features<-na.omit(test_data_df)
test_index_list_vec<-levels(test_features$id)
test_index_list_len<-length(test_index_list_vec)

for (i in 1:length(test_features$id)) {
  #get true index of one time series
  index_char_vec=str_extract_all(test_index_list_vec[i],"[0-9]")[[1]]
  index_str=str_c(index_char_vec)
  str_link=paste(index_str[1:length(index_str)],collapse="")
  j<-as.numeric(str_link)
  period_type<-strsplit(test_index_list_vec[i],"")[[1]][1]
  if(period_type=='Y'){
    index_num<-j
  }else if(period_type=='Q'){
    index_num<-j+23000
  }else if(period_type=='M'){
    index_num<-j+47000
  }else if(period_type=='W'){
    index_num<-j+95000
  }else if(period_type=='D'){
    index_num<-j+95359
  }else if(period_type=='H'){
    index_num<-j+99586
  }
  #index_of_ts_temp=c(index_of_ts_temp,index_num)
  M4_test[[index_num]]$features<-test_features[i,-1]
}
# for(i in 1:index_list_len){
#   #add features accroding to index_list
#   M4_test[[index_list_vec[i]]]$features<-test_features[i,-1]
# }
tes_fe=M4_test[[1]]$ff

#########################test forecasting value#############
#file.of.testing.prediction.values="./m4_9_methods_1-10000/test/1-100000/prediction.value"
file.of.forecasts<-dir(file.of.testing.prediction.values)
fore.value.len<-length(file.of.forecasts)
for (i in 1:fore.value.len){
  #read csv mase
  forecast_of_ts<-read.csv(paste(file.of.testing.prediction.values,file.of.forecasts[i],sep = "/"))
  row.names(forecast_of_ts)<-forecast_of_ts[,2]
  forecasting_value<-forecast_of_ts[,-c(1,2)]
  #get true index of one time series
  #value_file_name<-file_forecasting[e]
  index_char_vec=str_extract_all(file.of.forecasts[i],"[0-9]")[[1]]
  index_str=str_c(index_char_vec)
  str_link=paste(index_str[1:length(index_str)],collapse="")
  j<-as.numeric(str_link)
  period_type<-strsplit(file.of.forecasts[i],"")[[1]][1]
  if(period_type=='Y'){
    index_num<-j
  }else if(period_type=='Q'){
    index_num<-j+23000
  }else if(period_type=='M'){
    index_num<-j+47000
  }else if(period_type=='W'){
    index_num<-j+95000
  }else if(period_type=='D'){
    index_num<-j+95359
  }else if(period_type=='H'){
    index_num<-j+99586
  }
  #index.of.train.data=c(index.of.train.data,int_index)
  #index_of_ts_temp=c(index_of_ts_temp,index_num)
  M4_test[[index_num]]$ff<-as.matrix(forecasting_value)
}


#################create testing data################
test_data <- create_feat_classif_problem(M4_test)
###################prediction#######################
preds <- predict_selection_ensemble(meta_model, test_data$data)
head(preds)
M4_test <- ensemble_forecast(preds, M4_test)
M4_test[[1]]$y_hat
#calculate mase of every time series
#M4_test<-calc_errors_new(M4_test)
smape_all_new<-c()
mase_all_new<-c()
#write predict result to csv
#owa
for(r in 1:100000){
  #print(r)
  smape_h<-smape_cal(M4_test[[r]]$xx,M4_test[[r]]$y_hat)
  smape_temp<-mean(smape_h)
  smape_all_new<-c(smape_all_new,smape_temp)
  
  mase_h<-mase_cal(M4_test[[r]]$x,M4_test[[r]]$xx,M4_test[[r]]$y_hat)
  #mase_temp<-tail(mase_h,1)
  mase_temp<-mean(mase_h)
  mase_all_new<-c(mase_all_new,mase_temp)
  #calculate owa
  #owa<-0.5*((smape_temp/smape_of_naive2) +  (mase_temp/mase_of_naive2))
  #owa_all_new<-c(owa_all_new,owa)
}
errorsnew<-matrix(data=NA,nrow=7,ncol=3)
errorsnew<-data.frame(errorsnew)
colnames(errorsnew)<-c("smape","mase")
#rolnames(errorsnew)<-c("Yearly","quarterly","monthly","weekly","daily","hourly","all")
errorsnew$mase<-c(mean(mase_all_new[1:23000]),mean(mase_all_new[23001:47000]),mean(mase_all_new[47001:95000]),mean(mase_all_new[95001:95359]),mean(mase_all_new[95360:99586]),mean(mase_all_new[99587:100000]),mean(mase_all_new))
errorsnew$smape<-c(mean(smape_all_new[1:23000]),mean(smape_all_new[23001:47000]),mean(smape_all_new[47001:95000]),mean(smape_all_new[95001:95359]),mean(smape_all_new[95360:99586]),mean(smape_all_new[99587:100000]),mean(smape_all_new))
errorsnew$owa<-c((mean(mase_all_new[1:23000])/3.974+mean(smape_all_new[1:23000])/16.342)/2,
                 (mean(mase_all_new[23001:47000])/1.371+mean(smape_all_new[23001:47000])/11.012)/2,
                 (mean(mase_all_new[47001:95000])/1.063+mean(smape_all_new[47001:95000])/14.427)/2,
                 (mean(mase_all_new[95001:95359])/2.777+mean(smape_all_new[95001:95359])/9.161)/2,
                 (mean(mase_all_new[95360:99586])/3.278+mean(smape_all_new[95360:99586])/3.045)/2,
                 (mean(mase_all_new[99587:100000])/2.395+mean(smape_all_new[99587:100000])/18.383)/2,
                 (mean(mase_all_new)/1.912+mean(smape_all_new)/13.564)/2)
write.csv(errorsnew,file="./M4accuracy/M4_resnet101.csv",row.names=TRUE,col.names=TRUE)


