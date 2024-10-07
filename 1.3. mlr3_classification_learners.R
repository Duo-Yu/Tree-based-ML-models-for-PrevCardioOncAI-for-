#-----------------------------------------
rm(list=ls())
library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3tuning)
library(pROC)

#---------------------
# 1. read training and testing data

data_imputed = read.csv(paste0(dir,'all_data_3835_1.csv'))
train_y = read.csv(paste0(data_dir,'MCW_train_data_label.csv'))
test_y = read.csv(paste0(data_dir,'MCW_test_data_label.csv'))

data_imputed$race.black = rep(0, nrow(data_imputed))
data_imputed$race.black[which(data_imputed$race == 'black')] = 1
data_imputed$race.other = rep(0, nrow(data_imputed))
data_imputed$race.other[which(data_imputed$race == 'other')] = 1
data_imputed= data_imputed[,-2]
data_imputed$race.black = as.factor(data_imputed$race.black)
data_imputed$race.other = as.factor(data_imputed$race.other)
data_imputed$age = as.numeric(data_imputed$age)
re_name = c('age', 'race.black','race.other', colnames(data_imputed)[2:87])
data_imputed = data_imputed[,re_name]

data_imputed1 = data_imputed
for (i in 2:17){
  data_imputed1[,i] = as.factor( data_imputed1[,i])
}

#---------------
# 1.1 read validation data, only has female?
data_imputed_valid = read.csv(paste0(dir,'valid_data_329.csv'))
label_valid = read.csv(paste0(dir,'valid_label_329.csv'))

data_imputed_valid$race.black = rep(0, nrow(data_imputed_valid))
data_imputed_valid$race.black[which(data_imputed_valid$race == 'black')] = 1
data_imputed_valid$race.other = rep(0, nrow(data_imputed_valid))
data_imputed_valid$race.other[which(data_imputed_valid$race == 'other')] = 1
data_imputed_valid = data_imputed_valid[,-2]
data_imputed_valid$race.black = as.factor(data_imputed_valid$race.black)
data_imputed_valid$race.other = as.factor(data_imputed_valid$race.other)
data_imputed_valid$age = as.numeric(data_imputed_valid$age)
re_name = c('age', 'race.black','race.other', colnames(data_imputed_valid)[2:86])
data_imputed_valid = data_imputed_valid[,re_name]

data_imputed_valid_1 = data_imputed_valid
for (i in 2:17){
  data_imputed_valid_1[,i] = as.factor( data_imputed_valid_1[,i])
}
levels(data_imputed_valid_1$male) = c(0, 1)
#---------------------
# 2. create mlr classification task for most of the learners

# input parameter
# param 1: outcome index, from 1-4
# param 2: seed, 5 diffrent seeds
# output of AUC for test set and validation set

seed.pool = c(100, 200, 300, 400, 500)

for (y_index in 1: 4){
  #y_index = 1
 
    #s = seed.pool[1]
    Yvar = colnames(train_y)[y_index]
    Xvars = colnames(data_imputed)[1:88]

data.train <- cbind(train_y[,y_index], data_imputed1[data_imputed1$train_index==1,Xvars])
colnames(data.train)[1] = 'Y'
data.train$Y= as.factor(data.train$Y)

data.test <- cbind(test_y[,y_index], data_imputed1[data_imputed1$train_index==0,Xvars])
colnames(data.test)[1] = 'Y'
data.test$Y= as.factor(data.test$Y)

data.valid <-cbind(label_valid[,y_index], data_imputed_valid_1[,Xvars])
colnames(data.valid)[1]= 'Y'
data.valid$Y= as.factor(data.valid$Y)

task_classif_train = TaskClassif$new(id = 'PrevCardio_train', backend = data.train, target = 'Y' )
task_classif_test =  TaskClassif$new(id = 'PrevCardio_test',  backend = data.test, target = 'Y' )
task_classif_valid =  TaskClassif$new(id = 'PrevCardio_valid', backend = data.valid, target = 'Y' )

# XGboost does not support factor feature, create new task for XGBoost
data_imputed2 = data_imputed
for (i in 2:17){
  data_imputed2[,i] = as.numeric(as.character(data_imputed2[,i]))
}
data.train_xg <- cbind(train_y[,y_index], data_imputed2[data_imputed2$train_index==1,Xvars])
colnames(data.train_xg)[1] = 'Y'
data.train_xg$Y= as.factor(data.train_xg$Y)

data.test_xg <- cbind(test_y[,y_index], data_imputed2[data_imputed2$train_index==0,Xvars])
colnames(data.test_xg)[1] = 'Y'
data.test_xg$Y= as.factor(data.test_xg$Y)

data_imputed_valid_2 = data_imputed_valid
for (i in 2:17){
  data_imputed_valid_2[,i] = as.numeric(as.character(data_imputed_valid_2[,i]))
}
data.valid_xg <- cbind(label_valid[,y_index], data_imputed_valid_2[,Xvars])
colnames(data.valid_xg)[1] = 'Y'
data.valid_xg$Y= as.factor(data.valid_xg$Y)

task_classif_train_xg = TaskClassif$new(id = 'PrevCardio_train_xg', backend = data.train_xg, target = 'Y' )
task_classif_test_xg =  TaskClassif$new(id = 'PrevCardio_test_xg', backend = data.test_xg, target = 'Y' )
task_classif_valid_xg =  TaskClassif$new(id = 'PrevCardio_valid_xg', backend = data.valid_xg, target = 'Y' )

#-------------------------------
# 3. set up nested cross-validation 
inner_resampling <- rsmp('cv',folds = 5) # for hyper-parameter tuning
tuner <- tnr("grid_search", resolution = 50)
outer_resampling <- rsmp("cv", folds = 10) # for validating model performance

#-------------------------------------------------------------------------------
# 4. implemente various classifiers 

#------
# 4.1. regularized logistic regression 
logit.auc.test = c()
logit.auc.valid = c()
library(glmnet) 
learner_logistic = lrn("classif.glmnet", predict_type = 'prob')
#learner_logistic$param_set
#learner_logistic$train(task_classif_train)
#predictions = learner_logistic$predict(task_classif_test)
#roc_obj <- roc(predictions$truth, predictions$prob[, 2])
#auc(roc_obj) # default learner auc 0.8033

search_space_logit = ps(
  alpha = p_dbl( lower = 0, upper = 1),
  lambda = p_dbl(lower = 0.00001, upper = 10)
)

auto_tuner <- AutoTuner$new(
  learner = learner_logistic,
  resampling = inner_resampling,
  measure = msr("classif.auc"),
  search_space = search_space_logit,
  terminator = trm("evals", n_evals = 100),
  tuner = tuner
)

for(s in seed.pool){
set.seed(s)
rr.logit  =  resample(task_classif_train_xg, auto_tuner, outer_resampling, store_models = TRUE)
inner_tune = extract_inner_tuning_results(rr.logit)
outer_eval = rr.logit$score()
best_iteration = outer_eval[outer_eval$classif.ce == min(outer_eval$classif.ce ),iteration]
best_model = inner_tune[inner_tune$iteration == best_iteration[1],  ]
best_model_params = best_model$learner_param_vals

learner_logistic$param_set$values$alpha = best_model$alpha
learner_logistic$param_set$values$lambda = best_model$lambda

# train and evaluate on test set
learner_logistic$train(task_classif_train_xg)
predictions = learner_logistic$predict(task_classif_test_xg)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj)

# valid
pred_valid = learner_logistic$predict(task_classif_valid_xg)
roc_obj_valid <- roc(pred_valid$truth, pred_valid$prob[, 2])
auc(roc_obj_valid)

logit.auc.test = c(logit.auc.test, auc(roc_obj))
logit.auc.valid = c(logit.auc.valid,auc(roc_obj_valid))
}
logit.output = data.frame(
  logit.test = logit.auc.test,
  logit.valid = logit.auc.valid
)
write.csv(logit.output,file= paste0(result_dir,'logit_',colnames(test_y)[y_index],'.csv'), row.names = F)
#-------------------------------------------------------------------------------
# 4.2. random forests src
learner_rfsrc = lrn("classif.rfsrc",predict_type ='prob')
learner_rfsrc$param_set
search_space_rfsrc = ps(
    ntree = p_int(lower = 500, upper = 2000),
    nodesize = p_int(lower = 5, upper = 25),
    mtry = p_int(lower = 1, upper = 10),
    splitrule = p_fct(levels = c('gini'))
  )


auto_tuner <- AutoTuner$new(
  learner = learner_rfsrc,
  resampling = inner_resampling,
  measure = msr("classif.auc"),
  search_space = search_space_rfsrc,
  terminator = trm("perf_reached", level = 0.001),
  tuner = tnr('random_search')
)

rf.auc.test = c()
rf.auc.valid = c()
for(s in seed.pool){
set.seed(s)
rr.rfsrc  =  resample(task_classif_train, auto_tuner, outer_resampling, store_models = TRUE)

inner_tune = extract_inner_tuning_results(rr.rfsrc)
outer_eval = rr.rfsrc$score()
best_iteration = outer_eval[outer_eval$classif.ce == min(outer_eval$classif.ce ),iteration]
best_model = inner_tune[inner_tune$iteration == best_iteration[1],  ]
best_model_params = best_model$learner_param_vals

learner_rfsrc$param_set$values$ntree = best_model$ntre
learner_rfsrc$param_set$values$nodesize = best_model$nodesize
learner_rfsrc$param_set$values$mtry = best_model$mtry
learner_rfsrc$param_set$values$splitrule = best_model$splitrule

# train and evaluate on test set
learner_rfsrc$train(task_classif_train) 
predictions = learner_rfsrc$predict(task_classif_test)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj)   # default learner auc is 0.8013

# valid
pred_valid = learner_rfsrc$predict(task_classif_valid)
roc_obj_valid <- roc(pred_valid$truth, pred_valid$prob[, 2])
auc(roc_obj_valid)

rf.auc.test = c(rf.auc.test, auc(roc_obj))
rf.auc.valid = c(rf.auc.valid,auc(roc_obj_valid))
}
rf.output = data.frame(
  rf.test = rf.auc.test,
  rf.valid = rf.auc.valid
)

write.csv(rf.output,file= paste0(result_dir,'rf_',colnames(test_y)[y_index],'.csv'), row.names = F)


#-------------------------------------------------------------------------------
# 4.3. MLP broke R session
#learner_mlp = lrn('classif.multilayer_perceptron',predict_type ='prob')
# using default learner to predict
#learner_mlp$train(task_classif_train) 
#predictions = learner_mlp$predict(task_classif_test)
#roc_obj <- roc(predictions$truth, predictions$prob[, 2])
#auc(roc_obj)  
#-------------------------------------------------------------------------------
# 4.4. BART
learner_bart = lrn('classif.bart',predict_type ='prob')
learner_bart$train(task_classif_train) 
predictions = learner_bart$predict(task_classif_test)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj)   # 0.803
learner_bart$param_set

search_space_bart = ps(
  ntree = p_int(lower = 500, upper = 2000),  # number of trees at each iteration
  k = p_dbl(lower = 2, upper = 10),   # check dbarts reference at page 4
  power = p_dbl(lower = 1, upper = 5),
  base = p_dbl(lower = 0, upper = 1),
  nskip = p_int(lower = 50, upper = 200),  # number of bur-in
  ndpost = p_int(lower = 500, upper = 2000) # number of iterations
)


auto_tuner <- AutoTuner$new(
  learner = learner_bart,
  resampling = inner_resampling,
  measure = msr("classif.auc"),
  search_space = search_space_bart,
  terminator = trm("perf_reached", level = 0.001),
  tuner = tnr('random_search')
)

bart.auc.test = c()
bart.auc.valid = c()

for(s in seed.pool){
set.seed(s)
rr.bart  =  resample(task_classif_train, auto_tuner, outer_resampling, store_models = TRUE)

inner_tune = extract_inner_tuning_results(rr.bart)
outer_eval = rr.bart$score()
best_iteration = outer_eval[outer_eval$classif.ce == min(outer_eval$classif.ce ),iteration]
best_model = inner_tune[inner_tune$iteration == best_iteration[1],  ]
best_model_params = best_model$learner_param_vals

learner_bart$param_set$values$ntree = best_model$ntree
learner_bart$param_set$values$k = best_model$k
learner_bart$param_set$values$power = best_model$power
learner_bart$param_set$values$base = best_model$base
learner_bart$param_set$values$nskip = best_model$nskip
learner_bart$param_set$values$ndpost = best_model$ndpost

# train and evaluate on test set
learner_bart$train(task_classif_train) 

predictions = learner_bart$predict(task_classif_test)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj) 

# valid
pred_valid = learner_bart$predict(task_classif_valid)
roc_obj_valid <- roc(pred_valid$truth, pred_valid$prob[, 2])
auc(roc_obj_valid)

bart.auc.test = c(bart.auc.test, auc(roc_obj))
bart.auc.valid = c(bart.auc.valid,auc(roc_obj_valid))
}

bart.output = data.frame(
  bart.test = bart.auc.test,
  bart.valid = bart.auc.valid
)

write.csv(bart.output,file= paste0(result_dir,'bart_',colnames(test_y)[y_index],'.csv'), row.names = F)


#-------------------------------------------------------------------------------
# 4.5. XGboost
#----------

learner_xgboost = lrn("classif.xgboost",predict_type ='prob')
learner_xgboost$train(task_classif_train_xg) 
predictions = learner_xgboost$predict(task_classif_test_xg)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj)  # 0.7388

# based on paper "Bischl, Bernd, Martin Binder, Michel Lang, 
#Tobias Pielok, Jakob Richter, Stefan Coors, Janek Thomas, 
#et al. 2021. "Hyperparameter Optimization: Foundations, 
# Algorithms, Best Practices and Open Challenges."" Table 6, 8 hyperparameters
learner_xgboost$param_set

search_space_xgboost = ps(
  eta = p_dbl(lower = 0, upper = 1),  
  nrounds = p_int(lower = 1, upper = 5000),  
  max_depth = p_int(lower = 1, upper = 20),
  colsample_bytree = p_dbl(lower = 0.1, upper = 1),
  colsample_bylevel = p_dbl(lower = 0.1, upper = 1),  
  lambda = p_dbl(lower = 0, upper = 10),
  alpha = p_dbl(lower = 0, upper = 10),
  subsample = p_dbl(lower = 0.1, upper = 1)
)


auto_tuner <- AutoTuner$new(
  learner = learner_xgboost,
  resampling = inner_resampling,
  measure = msr("classif.auc"),
  search_space = search_space_xgboost,
  terminator = trm("perf_reached", level = 0.001),
  tuner = tnr('random_search')
)

xgboost.auc.test = c()
xgboost.auc.valid = c()

for(s in seed.pool){
set.seed(s)
rr.xgboost  =  resample(task_classif_train_xg, auto_tuner, outer_resampling, store_models = TRUE)

inner_tune = extract_inner_tuning_results(rr.xgboost)
outer_eval = rr.xgboost$score()
best_iteration = outer_eval[outer_eval$classif.ce == min(outer_eval$classif.ce ),iteration]
best_model = inner_tune[inner_tune$iteration == best_iteration[1],  ]
best_model_params = best_model$learner_param_vals

learner_xgboost$param_set$values$eta = best_model$eta
learner_xgboost$param_set$values$nrounds = best_model$nrounds
learner_xgboost$param_set$values$max_depth = best_model$max_depth
learner_xgboost$param_set$values$colsample_bytree = best_model$colsample_bytree
learner_xgboost$param_set$values$colsample_bylevel = best_model$colsample_bylevel
learner_xgboost$param_set$values$lambda = best_model$lambda
learner_xgboost$param_set$values$alpha = best_model$alpha
learner_xgboost$param_set$values$subsample = best_model$subsample

# train and evaluate on test set
learner_xgboost$train(task_classif_train_xg) 
predictions = learner_xgboost$predict(task_classif_test_xg)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj) # 0.7885

# valid
pred_valid = learner_xgboost$predict(task_classif_valid_xg)
roc_obj_valid <- roc(pred_valid$truth, pred_valid$prob[, 2])
auc(roc_obj_valid)

xgboost.auc.test = c(xgboost.auc.test, auc(roc_obj))
xgboost.auc.valid = c(xgboost.auc.valid,auc(roc_obj_valid))
}

xgboost.output = data.frame(
  xgboost.test = xgboost.auc.test,
  xgboost.valid = xgboost.auc.valid
)

write.csv(xgboost.output,file= paste0(result_dir,'xgboost_',colnames(test_y)[y_index],'.csv'), row.names = F)


#-------------------------------------------------------------------------------
# 4.6. lightGBM
learner_lightgbm = lrn("classif.lightgbm",predict_type ='prob')
learner_lightgbm$train(task_classif_train) 
predictions = learner_lightgbm$predict(task_classif_test)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj) # 0.7867
learner_lightgbm$param_set

# based on paper "Revisiting Gradient Boosting-Based Approaches for
# Learning Imbalanced Data: A Case of Anomaly Detection on Power Grids"
# 11 hyperparameters:  maximum bin, maximum depth, minimum data in leaf, learning rate,
# lambda 𝑙1, lambda 𝑙2, tree size, feature fraction, 
# bagging fraction, path smoothing, and minimum gain to split.

search_space_lightgbm = ps(
  max_bin = p_int(lower = 100, upper = 250),  
  max_depth = p_int(lower = 1, upper = 15),  
  min_data_in_leaf= p_int(lower = 100, upper = 1000),
  learning_rate = p_dbl(lower = 0.1, upper = 0.3),
  lambda_l1 = p_dbl(lower = 0, upper = 100),  
  lambda_l2 = p_dbl(lower = 0, upper = 100),
  feature_fraction = p_fct(levels =c(0.5, 0.9)),
  bagging_fraction = p_fct(levels =c(0.5, 0.9)),
  path_smooth = p_dbl(lower = 0.000001, upper = 0.0001),
  min_gain_to_split = p_int(lower = 1, upper = 15)
)


auto_tuner <- AutoTuner$new(
  learner = learner_lightgbm,
  resampling = inner_resampling,
  measure = msr("classif.auc"),
  search_space = search_space_lightgbm,
  terminator = trm("perf_reached", level = 0.001),
  tuner = tnr('random_search'),
  #tuner = tuner
)

lightgbm.auc.test = c()
lightgbm.auc.valid = c()
for (s in seed.pool){
set.seed(s)
rr.lightgbm  =  resample(task_classif_train, auto_tuner, outer_resampling, store_models = TRUE)

inner_tune = extract_inner_tuning_results(rr.lightgbm)
outer_eval = rr.lightgbm$score()
best_iteration = outer_eval[outer_eval$classif.ce == min(outer_eval$classif.ce ),iteration]
best_model = inner_tune[inner_tune$iteration == best_iteration[1],  ]
best_model_params = best_model$learner_param_vals


# error here
learner_lightgbm$param_set$values$max_depth = best_model$max_depth
learner_lightgbm$param_set$values$min_data_in_leaf = best_model$min_data_in_leaf
learner_lightgbm$param_set$values$learning_rate = best_model$learning_rate
learner_lightgbm$param_set$values$lambda_l1 = best_model$lambda_l1
learner_lightgbm$param_set$values$lambda_l2 = best_model$lambda_l2
learner_lightgbm$param_set$values$feature_fraction = as.numeric(best_model$feature_fraction)
learner_lightgbm$param_set$values$bagging_fraction = as.numeric(best_model$bagging_fraction)
learner_lightgbm$param_set$values$path_smooth = best_model$path_smooth
learner_lightgbm$param_set$values$min_gain_to_split = best_model$min_gain_to_split

# train and evaluate on test set
learner_lightgbm$train(task_classif_train)  
predictions = learner_lightgbm$predict(task_classif_test)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj) # 0.7633

# valid
pred_valid = learner_lightgbm$predict(task_classif_valid)
roc_obj_valid <- roc(pred_valid$truth, pred_valid$prob[, 2])
auc(roc_obj_valid)

lightgbm.auc.test = c(lightgbm.auc.test, auc(roc_obj))
lightgbm.auc.valid = c(lightgbm.auc.valid,auc(roc_obj_valid))
}

lightgbm.output = data.frame(
  lightgbm.test = lightgbm.auc.test,
  lightgbm.valid = lightgbm.auc.valid
)

write.csv(lightgbm.output,file= paste0(result_dir,'lightgbm_',colnames(test_y)[y_index],'.csv'), row.names = F)


#-------------------------------------------------------------------------------
# 4.7. CatBoost
learner_catboost = lrn("classif.catboost",predict_type ='prob')
learner_catboost$train(task_classif_train) 
predictions = learner_catboost$predict(task_classif_test)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj) # 0.8049

# based on paper "Revisiting Gradient Boosting-Based Approaches for
# Learning Imbalanced Data: A Case of Anomaly Detection on Power Grids"
search_space_catboost = ps(
  depth = p_int(lower = 1, upper = 10),  
  learning_rate = p_fct(levels = c(0.03, 0.001, 0.01, 0.1, 0.2, 0.3)),  
  l2_leaf_reg= p_fct(levels = c(1, 3, 5, 10, 100)),
  border_count = p_fct(levels = c(5, 10, 20, 30, 50, 100, 200)),
  boosting_type = p_fct(levels = c('Ordered','Plain'))
)



auto_tuner <- AutoTuner$new(
  learner = learner_catboost,
  resampling = inner_resampling,
  measure = msr("classif.auc"),
  search_space = search_space_catboost,
  terminator = trm("perf_reached", level = 0.001),
  tuner = tnr('random_search')
)

catboost.auc.test = c()
catboost.auc.valid = c()
for (s in seed.pool){
set.seed(s)
rr.catboost  =  resample(task_classif_train, auto_tuner, outer_resampling, store_models = TRUE)

inner_tune = extract_inner_tuning_results(rr.catboost)
outer_eval = rr.catboost$score()
best_iteration = outer_eval[outer_eval$classif.ce == min(outer_eval$classif.ce ),iteration]
best_model = inner_tune[inner_tune$iteration == best_iteration[1],  ]
best_model_params = best_model$learner_param_vals

learner_catboost$param_set$values$depth = best_model$depth
learner_catboost$param_set$values$learning_rate = as.numeric(best_model$learning_rate)
learner_catboost$param_set$values$l2_leaf_reg = as.numeric(best_model$l2_leaf_reg)
learner_catboost$param_set$values$border_count = as.numeric(best_model$border_count)
learner_catboost$param_set$values$boosting_type = best_model$boosting_type

# train and evaluate on test set
learner_catboost$train(task_classif_train)  
predictions = learner_catboost$predict(task_classif_test)
roc_obj <- roc(predictions$truth, predictions$prob[, 2])
auc(roc_obj) # 0.7885

# valid
pred_valid = learner_catboost$predict(task_classif_valid)
roc_obj_valid <- roc(pred_valid$truth, pred_valid$prob[, 2])
auc(roc_obj_valid)
catboost.auc.test = c(catboost.auc.test, auc(roc_obj))
catboost.auc.valid = c(catboost.auc.valid,auc(roc_obj_valid))
}
catboost.output = data.frame(
  catboost.test = catboost.auc.test,
  catboost.valid = catboost.auc.valid
)

write.csv(catboost.output,file= paste0(result_dir,'catboost_',colnames(test_y)[y_index],'.csv'), row.names = F)
}

