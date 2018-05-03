# STAT406, UBC, Fall 2017, Final Project
# Author: DUONG, BANG CHI 

#### Load libraries and Source other files if needed ####
sourceFunctions <- "C:/Users/DBC/OneDrive/R_myFunctions"
source(paste(sourceFunctions, "Graphs.R", sep = "/"))

# Packages and Source of functions ####
library(data.table)
# library(ResourceSelection) # for Hosmer-Lemeshow statistic
library(MASS) # for LDA
library(e1071) # for SVM
library(randomForest)
#library(adabag)
library(ada)
library(FastKNN)

#### STORAGE ####
#setwd("C:/Users/DBC/OneDrive/Courses/UBC/Fall2017/STAT406/STAT406_R/Project")
currentStorage <- paste(getwd(), sep = "/")

dir.create(path = paste(currentStorage, "fig", sep = "/"), showWarnings = TRUE)

# Load data and clean ####
mammo <- fread(paste(currentStorage, "mammographic", "mammographic.data", sep = "/"), na.strings = "?")
colnames(mammo) <- c("BIRAD", "Age", "Shape", "Margin", "Density", "Class")

mammo <- within(mammo, {
  BIRAD <- NULL
  Shape <- factor(ifelse(Shape==1, "round",
                         ifelse(Shape==2, "oval",
                                ifelse(Shape==3, "lobular",
                                       ifelse(Shape==4, "irregular", NA)))),
                  levels = c("round","oval","lobular","irregular"))
  Margin <- factor(ifelse(Margin==1, "circumscribed",
                          ifelse(Margin==2, "microlobulated",
                                 ifelse(Margin==3, "obscured",
                                        ifelse(Margin==4, "ill_defined", 
                                               ifelse(Margin==5, "spiculated", NA))))),
                   levels = c("circumscribed","microlobulated","obscured","ill_defined","spiculated"))
  Density <- factor(ifelse(Density==1, "high",
                           ifelse(Density==2, "iso",
                                  ifelse(Density==3, "low",
                                         ifelse(Density==4, "fat_containing", NA)))),
                    levels = c("high","iso","low","fat_containing"))
  Class <- factor(ifelse(Class==0,"Benign",
                         ifelse(Class==1,"Malignant",NA)))
})

myData <- na.omit(mammo)

# Some plots ----
png(filename = "MarginRelationship.png", width = 10, height = 7, units = "in", res = 300)
par(mfrow=c(2,2))

boxplot(Age ~ Class, data = myData, ylab = "Age", xlab = "Severity", main = "Age vs Severity")

for (i in 2:4) {
  mosaicplot(myData$Class ~ unlist(myData[,.SD,.SDcols=names(myData)[i]]),
             ylab = names(myData)[i], xlab = "Severity",
             main = paste(names(myData)[i], " vs Severity" ))
}

dev.off()

par(mfrow=c(1,1))

Num1_Cat3(dat = subset(myData, select = c(Shape,Age,Margin,Class)), 
          PlotTitle = "Overall trend of data", filename = "Overall.png")


# Cross_Validation ####

# K-CV ####
Kfold_CV <- function(data, response = "response", k_fold = 5, N = 10, seed = 1, predictors = "A + B + C") {
  
  # Set seed
  set.seed(seed)
  # Number of observations
  n <- nrow(data)
  # Index each fold, make a sequence of them
  fold_ind <- ( (1:n) %% k_fold ) +1 #rep(1:k_fold, length.out = n)
  # Models considered
  models <- c("logit", "lda", "qda", "svm.linear", "svm.radial",
              "random.forest", "boost.1", "boost.2", "boost.3")
  
  # Arrays to store MSPE
  mspe <- list()
  
  for (model_i in 1:length(models)) {
    mspe[[model_i]] <- rep(0, N) 
  }
  names(mspe) <- models
  
  count_run <- 1
  for(run in 1:N) {
    
    cat("N =", count_run, "\n")
    
    fold_ind <- sample(fold_ind)
    
    # Array to store predictions
    prediction <- list()
    for (model_i in 1:length(models)) {
      prediction[[model_i]] <- rep(0, n)
    }
    names(prediction) <- models
    
    count_fold <- 1
    for(fold in 1:k_fold) {
      # Define training and validation sets
      train <- data[fold_ind != fold, ]
      valid <- data[fold_ind == fold, ]
      
      # Parametric models ----
      # Logistic model
      fit.logit <- glm(data = train,
                       as.formula(paste(response, "~", predictors, sep = "")),
                       family = binomial)
      
      # LDA
      fit.lda <- lda(data = train, as.formula(paste(response, "~", predictors, sep = "")))
      
      # QDA
      # jitter the numeric predictors to prevent rank deficiency (multicolinearity)
      train.qda <- copy(train)
      for (j in names(data)[sapply(data,is.numeric)]) {
        set(train.qda, j = j, value = jitter(train.qda[[j]]))
      }
      fit.qda <- qda(data = train.qda, as.formula(paste(response, "~", predictors, sep = "")))
      
      # SVM - Linear
      fit.svm.linear <- svm(data = train, as.formula(paste(response, "~", predictors, sep = "")),
                            kernel = "linear", cost = 10, scale = FALSE)
      
      # SVM - Radial
      fit.svm.radial <- svm(data = train, as.formula(paste(response, "~", predictors, sep = "")),
                            kernel = "radial", cost = 10, scale = FALSE)
      
      # Non-parametric models ----
      # Random Forest, ntree + mtry
      fit.rf <- randomForest(data = train, as.formula(paste(response, "~", predictors, sep = "")), ntree = 500)
      
      # Boosting - Stump/one split
      one.split <- rpart.control(cp = -1, maxdepth = 1, minsplit = 0, xval = 0)
      fit.bo.1s <- ada(data = train, as.formula(paste(response, "~", predictors, sep = "")),
                       iter = 200, control = one.split)
      
      # Boosting - two splits
      two.split <- rpart.control(cp = -1, maxdepth = 2, minsplit = 0, xval = 0)
      fit.bo.2s <- ada(data = train, as.formula(paste(response, "~", predictors, sep = "")),
                       iter = 200, control = two.split)
      
      # Boosting - three splits
      three.split <- rpart.control(cp = -1, maxdepth = 3, minsplit = 0, xval = 0)
      fit.bo.3s <- ada(data = train, as.formula(paste(response, "~", predictors, sep = "")),
                       iter = 200, control = three.split)
      
      # Store all models except KNN into a list
      fits <- list(fit.logit, fit.lda, fit.qda, fit.svm.linear, fit.svm.radial,
                   fit.rf, fit.bo.1s, fit.bo.2s, fit.bo.3s)
      names(fits) <- models
      
      # Predictions ----
      prediction[["logit"]][ fold_ind == fold ] <- ifelse(
        predict(fits[["logit"]], newdata = valid, type = "response") < 0.5, "Benign", "Malignant"
      )
      
      for (k in c("lda", "qda")) {
        predTab <- as.data.table(predict(fits[[k]], newdata = valid)$posterior)
        prediction[[k]][ fold_ind == fold ] <- names(predTab)[ max.col(predTab, ties.method = "first") ]
      }
      
      for (k in c("svm.linear", "svm.radial", "random.forest", "boost.1", "boost.2", "boost.3")) {
        prediction[[k]][ fold_ind == fold ] <- as.character(predict(fits[[k]], newdata = valid))
      }
      
      cat("Fold =", count_fold, "\n")
      count_fold = count_fold + 1
    }
    
    # MSPE
    for (model_i in 1:length(mspe)) {
      mspe[[model_i]][run] <- mean( prediction[[model_i]] != data[[response]] )
    }
    
    
    count_run = count_run + 1
  }
  cat("DONE!\n")
  return(mspe)
}

# KCV: without density ----
system.time(
  result.KCV <- Kfold_CV(data = myData, response = "Class", k_fold = 5, N = 50, seed = 2017, predictors = "Age + Shape + Margin")
)

# user     system    elapsed 
# 2260.57    2.78    2343.33 

# Save result
saveRDS(result.KCV, "result_KCV.rds")



# KCV.2 : with density as a predictor ----
system.time(
  result.KCV.2 <- Kfold_CV(data = myData, response = "Class", k_fold = 5, N = 50, seed = 2017,
                           predictors = "Age + Shape + Margin + Density",
                           predictors.qda = "Age.jitter + Shape + Margin + Density")
)

saveRDS(result.KCV.2, file = "result_full.rds")

result.KCV <- readRDS("result_KCV.rds")

# Comparison with/without density ----
par(mfrow=c(2,1))
boxplot(result.KCV)
boxplot(result.KCV.2)

View(rbind(colMeans(as.data.table(result.KCV)),colMeans(as.data.table(result.KCV.2))))


# KNN ####
Kfold_CV_knn <- function(data, response = "response", k_fold = 5, N = 10, seed = 1, models = c("knn.1", "knn.3", ...)) {
  require(cluster)
  require(FastKNN)
  # Set seed
  set.seed(seed)
  # Number of observations
  n <- nrow(data)
  # Index each fold, make a sequence of them
  fold_ind <- ( (1:n) %% k_fold ) +1 #rep(1:k_fold, length.out = n)
  # Models considered
  models <- models
  
  # Arrays to store MSPE
  mspe <- list()
  
  for (model_i in 1:length(models)) {
    mspe[[model_i]] <- rep(0, N) 
  }
  names(mspe) <- models
  
  count_run <- 1
  for(run in 1:N) {
    
    cat("Run: N =", count_run, "\n")
    
    fold_ind <- sample(fold_ind)
    
    # Array to store predictions
    prediction <- list()
    for (model_i in 1:length(models)) {
      prediction[[model_i]] <- rep(0, n)
    }
    names(prediction) <- models
    
    count_fold <- 1
    for(fold in 1:k_fold) {
      # Define training and validation sets
      train <- data[fold_ind != fold, .SD, .SDcols = setdiff(names(data), response)]
      valid <- data[fold_ind == fold, .SD, .SDcols = setdiff(names(data), response)]
      train.response <- as.matrix(data[fold_ind != fold, .SD, .SDcols = response])
      
      # Predictions on KNN ----
      
      # Put the rows in each set into a list
      train.rowList <- split(train, seq(nrow(train)))
      valid.rowList <- split(valid, seq(nrow(valid)))
      
      distance <- list()
      
      for (row_i in 1:nrow(valid)) {
        
        distance[[row_i]] <- lapply(train.rowList, function(train.row) {
          
          daisy(rbind(train.row, valid.rowList[[row_i]]), metric = "gower")[1]
          
        })
      }
      
      distMatrix <- t(matrix(unlist(distance), nrow=nrow(train), ncol=nrow(valid)))
      
      for (model in models) {
        k_val <- as.numeric(unlist(strsplit(model, split = ".", fixed = TRUE))[2])
        prediction[[model]][ fold_ind == fold ] <- knn_test_function(dataset = train, test = valid,
                                                                 distance = distMatrix,
                                                                 labels = train.response,
                                                                 k = k_val)
      }
      
      cat("Fold =", count_fold, "\n")
      count_fold = count_fold + 1
    }
    
    # MSPE
    for (model_i in 1:length(mspe)) {
      mspe[[model_i]][run] <- mean( prediction[[model_i]] != data[[response]] )
    }
    
    count_run = count_run + 1
  }
  
  cat("DONE!\n")
  return(mspe)
}

##### ATTENTION!!!!! ######
##### I comment out this section because it takes a long time to train KNN, maybe there is a faster way to do!!! #####

# system.time(
#   result.knn.1 <- Kfold_CV_knn(data = myData[,.SD,.SDcols=c("Age","Shape","Margin","Class")], response = "Class", k_fold = 5, N = 20, seed = 2017, models = c("knn.1","knn.3","knn.5"))
# )
# 
# # user   system  elapsed 
# # 29009.23    22.00 31351.62 
# 
# saveRDS(result.knn.1, "result_kcv_knn_noDensity.rds")
# 
# result_knn <- readRDS("knn_withoutDensity.rds")
# 
# 
# system.time(
#   result.knn.2 <- Kfold_CV_knn(data = myData[,.SD,.SDcols=c("Age","Shape","Margin","Density","Class")], response = "Class", k_fold = 5, N = 20, seed = 2017, models = c("knn.1","knn.3","knn.5"))
# )
# saveRDS(result.knn.2, "result_kcv_knn_Density.rds")
# 
# 
# system.time(
#   result.9nn <- Kfold_CV_knn(data = myData[,.SD,.SDcols=c("Age","Shape","Margin","Class")], response = "Class", k_fold = 5, N = 20, seed = 2017, models = c("knn.9"))
# )
# saveRDS(result.9nn, "result_kcv_9nn_NoDensity.rds")
# 
# 
# system.time(
#   result.knn.11.13.15 <- Kfold_CV_knn(data = myData[,.SD,.SDcols=c("Age","Shape","Margin","Class")], response = "Class", k_fold = 5, N = 20, seed = 2017, models = c("knn.11","knn.13","knn.15"))
# )
# saveRDS(result.knn.11.13.15, "result_kcv_11_13_15nn_NoDensity.rds")
# 
# 
# # Plotting: only the case without Density
# # Load result ----
# result <- as.data.table(readRDS("result_KCV.rds"))
# result.knn.noDensity <- as.data.table(readRDS("result_kcv_knn_noDensity.rds"))
# 
# naList <- list()
# for (i in 1:3) {naList[[i]] <- rep(NA,30)}
# names(naList) <- c("knn.1","knn.3","knn.5")
# naList <- as.data.table(naList)
# 
# result.knn <- rbind(result.knn.noDensity, naList)
# 
# result <- cbind(result, result.knn)
# names(result) <- c("Logistic Regression", "LDA", "QDA",
#                         "SVM-Linear", "SVM-Radial",
#                         "Random Forest",
#                         "AdaBoost-1", "AdaBoost-2", "AdaBoost-3",
#                         "1-NN", "3-NN", "5-NN")
# View((colMeans(result, na.rm = TRUE)))

result <- cbind(result)
names(result) <- c("Logistic Regression", "LDA", "QDA",
                        "SVM-Linear", "SVM-Radial",
                        "Random Forest",
                        "AdaBoost-1", "AdaBoost-2", "AdaBoost-3")
View((colMeans(result, na.rm = TRUE)))

# Plot
result <- melt(data = as.data.table(result),
               measure.vars = c(1:12),
               variable.name = "Method", value.name = "Misclassification Error")

Boxplot(data = result,
        PlotTitle = "5-fold Cross Validation",
        filename = paste(currentStorage, "fig", "result_5foldCV", sep = "/"))