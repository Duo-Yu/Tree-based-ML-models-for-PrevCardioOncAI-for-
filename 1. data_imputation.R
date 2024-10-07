rm(list=ls())
library(missForest)
library(table1)

train_data = read.csv(paste0(data_dir,'MCW_train_data.csv'))
train_y = read.csv(paste0(data_dir,'MCW_train_data_label.csv'))
test_data = read.csv(paste0(data_dir,'MCW_test_data.csv'))
test_y = read.csv(paste0(data_dir,'MCW_test_data_label.csv'))



train_data$train_index = rep(1,nrow(train_data))
test_data$train_index = rep(0,nrow(test_data))
all_data = rbind(train_data,test_data)
table(all_data[,c('black','white','other')])

# remove column white, make white as the reference category of race
data_race = all_data[,2:4]
data_race$race = apply(data_race,1,function(x){
  ifelse(sum(x)==1,colnames(data_race)[which(x==1)],NA)})
all_data$race = data_race$race
all_data = all_data[,c('age','race',colnames(all_data)[5:90])]
all_data$race = as.factor(all_data$race)
# scale the data
all_data[,17:(ncol(all_data)-1)] = apply(all_data[,17:(ncol(all_data)-1)],2,scale)

# generate table 1, summarize data
#formula_my = paste0('~',colnames(train_data)[1])
#for (i in 2: (ncol(train_data)-1)){
#  if (i <= 17){
#    all_data[,i] = as.factor(all_data[,i])
#    formula_my = paste0(formula_my,'+', colnames(train_data)[i])
 # }else{
 #   formula_my = paste0(formula_my,'+', colnames(train_data)[i])
#  }}

#table1(formula(formula_my), data= all_data)

# random forests imputation
library(doParallel)
n_core <- detectCores()
registerDoParallel(cores=n_core) #### for paralel computing missForest
forest <- missForest(all_data, xtrue = NA, maxiter = 20,nodesize = c(5,5), 
                     ntree = 100, decreasing = FALSE, parallelize = 'forest')
imputed_data <- forest$ximp
#table1(formula(formula_my), data= imputed_data)

write.csv(imputed_data,file = 'all_data_3835_1.csv',row.names = F)

