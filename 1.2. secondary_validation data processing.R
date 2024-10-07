


data_valid = read.csv(paste0(data_dir1,'data_valid.csv'))
label_valid = read.csv(paste0(data_dir1,'label_valid.csv'))
#label_valid_1 = read.delim(paste0(data_dir1,'label_valid.txt'),header = F)

# Drop rows with > 60% missing values, columns with > 80%
missing_rate_row = apply(data_valid,1, function(x){sum(is.na(x))/ncol(data_valid) })
data_valid_1 = data_valid[missing_rate_row<= 0.6,]
label_valid_1 = label_valid[missing_rate_row <= 0.6,]
colnames(label_valid_1) = c('CVD','HF','AFib','CAD', 'MI','Stroke')
missing_rate_col = apply(data_valid_1, 2, function(x){sum(is.na(x))/nrow(data_valid_1)})
data_valid_2 = data_valid_1[, missing_rate_col <= 0.8]

# data_imputation 
train_data = read.csv(paste0(data_dir,'MCW_train_data.csv'))
train_y = read.csv(paste0(data_dir,'MCW_train_data_label.csv'))
data_valid_3 = data_valid_2[,colnames(train_data)]
label_valid_3 = label_valid_1[,colnames(train_y)]

data_race = data_valid_3[,2:4]
data_race$race = apply(data_race,1,function(x){
  ifelse(sum(x)==1,colnames(data_race)[which(x==1)],NA)})
data_valid_3$race = data_race$race
data_valid_3 = data_valid_3[,c('age','race',colnames(data_valid_3)[5:89])]
data_valid_3$race = as.factor(data_valid_3$race)
# scale the data
data_valid_3[,17:ncol(data_valid_3)] = apply(data_valid_3[,17:ncol(data_valid_3)],2,scale)

library(doParallel)
library(missForest)

n_core <- detectCores()
registerDoParallel(cores=n_core) #### for paralel computing missForest
forest <- missForest(data_valid_3, xtrue = NA, maxiter = 20,nodesize = c(5,5), 
                     ntree = 100, decreasing = FALSE, parallelize = 'forest')
imputed_data_valid <- forest$ximp
#table1(formula(formula_my), data= imputed_data)

write.csv(imputed_data_valid,file = 'valid_data_329.csv',row.names = F)
write.csv(label_valid_3,file = 'valid_label_329.csv',row.names = F)



