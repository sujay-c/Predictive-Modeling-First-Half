rm(list=ls()) #Removes every object from your environment

#Read data
df <- read.csv("games.csv")

#data-set consists of away and home team statistics
df1 <- df[,-c(14:20)] #drop the away team stats as only predicting number of points for home team
df <- df1[,-c(2,5,7)] #drop game id, visitor team id, home id (duplicate column)

#generating plots for variables of interest against y 
#to better understand relationships in data
plot(df$PTS_home,df$FG_PCT_home)
plot(df$PTS_home,df$FT_PCT_home)
plot(df$PTS_home,df$FG3_PCT_home)
plot(df$PTS_home,df$AST_home)
plot(df$PTS_home,df$REB_home)


#sample size
n = dim(df)[1] #23195 obsv

set.seed(1234) #ensures all our data is split the same 

#split into train (50%) / validation (25%) / test (25%)
sampleSizeTraining   <- floor(0.5 * n)
sampleSizeValidation <- floor(0.25 * n)
sampleSizeTest       <- floor(0.25 * n)

# Create the randomly-sampled indices for the dataframe. Use setdiff() to
# avoid overlapping subsets of indices.
indicesTraining    <- sort(sample(seq_len(n), size=sampleSizeTraining))
indicesNotTraining <- setdiff(seq_len(n), indicesTraining)
indicesValidation  <- sort(sample(indicesNotTraining, size=sampleSizeValidation))
indicesTest        <- setdiff(indicesNotTraining, indicesValidation)

# Finally, output the three data-frames for training, validation and test.
dfTraining   <- df[indicesTraining, ]
dfValidation <- df[indicesValidation, ]
dfTest       <- df[indicesTest, ]

#run our two approaches on the training data (dfTraining)
#validate each approach using kfold cv on the validation set (dfValidation)
#the last group will run the best approach on test data (dfTest) 

#if you only want the numeric values:
dfTraining <- dfTraining[,c(5:10)]
dfValidation <- dfValidation[,c(5:10)]
dfTest <- dfTest[,c(5:10)]

#testing using general linear regression 
lin <- lm(PTS_home ~ ., data = na.omit(dfTraining))
summary(lin) #traning RMSE of 8.008
pred.lm <- predict(lin, na.omit(dfValidation))
sqrt(mean((pred.lm - na.omit(dfValidation$PTS_home))^2)) #test RMSE of 8.0257

#creating matrixes for training and validation data
x = model.matrix(PTS_home~., data = dfTraining)
y = model.matrix(PTS_home~., data = dfValidation)

#Creating vectors for Y, getting rid of NA values 
PTS_home <- na.omit(dfTraining$PTS_home)
PTS_home_V <- na.omit(dfValidation$PTS_home)

#### RIDGE REGRESSION ####

library(glmnet)

#Setting up a variety of choices for lambda: this will implement the 
#function over a grid of values ranging from ?? = 1010 to ?? = 10???2
grid=10^seq(10,-2, length =100)

#setting up the ridge regression model WHAT IS THRESH??
ridge.mod = glmnet(x, PTS_home, alpha = 0, lambda = grid, thresh = 1e-12)

#using cross validation to identify the best lambda
crossValRidge = cv.glmnet(x, PTS_home, alpha = 0, lambda = grid, thresh = 1e-12)

#getting best lambda value using cross validation = 0.01
bestRidge = crossValRidge$lambda.min 

#using best lambda to apply our ridge regression to our validation data set 
ridge.pred = predict(crossValRidge, s = bestRidge, newx = y)

#our model produced a validation error of 8.025234
sqrt(mean((ridge.pred - PTS_home_V)^2))

#Coefficients Ridge regression model selected:
coef.R = predict(crossValRidge,type="coefficients",s=bestRidge)

'''
(Intercept)  -11.2299976
(Intercept)    .        
FG_PCT_home  120.3489824
FT_PCT_home   23.0587199
FG3_PCT_home  13.4844772
AST_home       0.6107641
REB_home       0.5104285
'''


#### LASSO REGRESSION ####


#setting up general lasso model 
lasso = glmnet(x, PTS_home, alpha = 1, lambda = grid, thresh = 1e-12)

#using cross validation to identify the best lambda
crossValLasso = cv.glmnet(x, PTS_home, alpha = 1, lambda = grid, thresh = 1e-12)

#getting best lambda value using cross validation = 0.01
bestLasso = crossValLasso$lambda.min

#using best lambda to apply our lasso regression to our validation data set 
lasso.pred = predict(crossValLasso, s = bestLasso, newx = y)

#our model produced an RMSE of 
sqrt(mean((lasso.pred - PTS_home_V)^2)) #8.025041

#Coefficients Lasso regression model selected: 
coef.L = predict(crossValLasso,type="coefficients",s=bestLasso)


'''
(Intercept)  -11.0623455
(Intercept)    .        
FG_PCT_home  120.3345352
FT_PCT_home   22.9744617
FG3_PCT_home  13.4158752
AST_home       0.6100014
REB_home       0.5091385
'''

#PCR

library(pls)
pcr.fit = pcr(PTS_home~., data = na.omit(dfTraining), scale = TRUE, validation = 'CV')
validationplot(pcr.fit,val.type="MSEP") #best M is 5
pcr.pred = predict(pcr.fit, na.omit(dfValidation), ncomp = 5)
sqrt(mean((pcr.pred - PTS_home_V)^2)) #8.025279


#PLS

pls.fit = plsr(PTS_home~., data = na.omit(dfTraining), scale = TRUE, validation = 'CV')
summary(pls.fit)
validationplot(pls.fit,val.type="MSEP") #best M is 3
pls.pred = predict(pls.fit, na.omit(dfValidation), ncomp = 3)
sqrt(mean((pls.pred - PTS_home_V )^2)) #8.032

### moving onto other methods.. 

#re-clarifying training/validation/test data so names don't get confused in preceding methods

dfTraining   <- df[indicesTraining, ]
dfValidation <- df[indicesValidation, ]
dfTest       <- df[indicesTest, ]

#Two models initially
dfTraining<-dfTraining[-c(1:4,11)]      # To remove the first 4 and the last column

Y_training<-dfTraining$PTS_home
dfTraining1<-dfTraining[,-1]
options(na.action="na.pass")
tmp<-model.matrix(~. ,dfTraining1)
dfTraining_2<-data.frame(Y_training,tmp)


null = lm(Y_training~1,na.action=na.exclude, data=dfTraining_2) #only has an intercept
full = glm(Y_training~.,na.action=na.exclude, data=dfTraining_2) #Has all the selected variables

regForward = step(null, #The most simple model
                  scope=formula(full),#Let us analyze all models until the full model
                  direction="forward", #Adding variables
                  k=log(length(indicesTraining))) #This is BIC

summary(regForward)

dfValidation<-dfValidation[-c(1:4,11)]
Y_Validation<-dfValidation$PTS_home
dfValidation_2<-dfValidation[,-1]

fwd.pred = predict(regForward, newdata = dfValidation_2 )
res<-data.frame((Y_Validation-fwd.pred)^2)
names(res)[1] <- "Errors"
sqrt(mean(res[, 'Errors'], na.rm = TRUE)) #RMSE is 8.025


#We'll implement the backward step regression model

regBack = step(full,                         #Starting with the full model
               direction="backward",         #And deleting variables
               k=log(length(indicesTraining)))

summary(regBack)

back.prd<- predict(regBack, newdata = dfValidation_2 )
res1<-data.frame((Y_Validation-back.prd)^2)
names(res1)[1] <- "Errors"
sqrt(mean(res1[, 'Errors'], na.rm = TRUE)) #RMSE is 8.025

#We'll implement the step regression in both the directions

regBoth =         step(null, #The most simple model
                       scope=formula(full), #The most complicated model
                       direction="both", #Add or delete variables
                       k=log(length(indicesTraining))) #This is BIC

summary(regBoth)

both.prd<- predict(regBoth, newdata = dfValidation_2 )
res2<-data.frame((Y_Validation-both.prd)^2)
names(res2)[1] <- "Errors"
sqrt(mean(res2[, 'Errors'], na.rm = TRUE)) #RMSE is 8.025

#################### RMSE Calculation for all 3 models on Test ################

# Now we'll use the test data set to find the RSME (Forward Step)

dfTest<-dfTest[-c(1:4,11)]
Y_Test<-dfTest$PTS_home
dfTest_2<-dfTest[,-1]
frd_test.pred<-predict(regForward, newdata = dfTest_2 )

res_frd.test<-data.frame((Y_Test-frd_test.pred)^2)
names(res_frd.test)[1] <- "Errors"
sqrt(mean(res_frd.test[, 'Errors'], na.rm = TRUE)) #RMSE is 7.83

# Now we'll use the test data to find the RSME (Backward Step)

bck_test.pred<-predict(regBack, newdata = dfTest_2)
res_bck.test<-data.frame((Y_Test - bck_test.pred)^2)
names(res_bck.test)[1] <- "Errors"
sqrt(mean(res_bck.test[, 'Errors'], na.rm = TRUE))  #RMSE is 7.83

# Now we'll use the test data to find the RSME(Both)

both_test.pred<-predict(regBoth, newdata = dfTest_2)
res_both.test<-data.frame((Y_Test - both_test.pred)^2)
names(res_both.test)[1] <- "Errors"
sqrt(mean(res_both.test[, 'Errors'], na.rm = TRUE)) #RMSE is 7.83

################# Mean of both Predicted and actual values ################

x<-data.frame(Y_Test)
names(x)[1] <- "Actuals"
mean(x[,'Actuals'],na.rm = TRUE)#102.0637

y<-data.frame(both_test.pred)
names(y)[1] <- "Predicted"
mean(y[,'Predicted'],na.rm = TRUE)#102.0746


############################# Boosting Method(Just a try) ########################################

library(gbm)
training_tmp<-na.omit(dfTraining)
boostfit = gbm(PTS_home~., #Formula (. means that all variables of df are included)
               
               data=training_tmp, #Data
               distribution='gaussian',
               interaction.depth=2, #Maximum depth of each tree
               n.trees=100, #Number of trees
               shrinkage=.2) #Learning rate

vsum=summary(boostfit,plotit=FALSE) #This will have the variable importance info



######################### Univariate Analysis#################################3
#Univariate analysis FG_PCT_home 

lm.fit<-lm(PTS_home~FG_PCT_home, data=dfTraining)
summary(lm.fit)
actuals<-dfValidation$PTS_home
test_tmp<-data.frame(FG_PCT_home=dfValidation$FG_PCT_home)
lm.pred <- predict(lm.fit,test_tmp)
sqrt(mean((lm.pred-actuals)^2,na.rm = TRUE))   #RMSE = 9.62

#Univariate analysis with FT_PCT_home
options(na.action="na.exclude")
lm.fit<-lm(PTS_home~FT_PCT_home, data=dfTraining)
summary(lm.fit)
actuals<-dfValidation$PTS_home
test_tmp<-data.frame(FT_PCT_home=dfValidation$FT_PCT_home)
lm.pred <- predict(lm.fit,test_tmp)
sqrt(mean((lm.pred-actuals)^2,na.rm = TRUE))  #RMSE = 12.77

#Univariate analysis with FG3_PCT_home

lm.fit<-lm(PTS_home~FG3_PCT_home, data=dfTraining)
summary(lm.fit)
actuals<-dfValidation$PTS_home
test_tmp<-data.frame(FG3_PCT_home=dfValidation$FG3_PCT_home)
lm.pred <- predict(lm.fit,test_tmp)
sqrt(mean((lm.pred-actuals)^2,na.rm = TRUE))  #RMSE = 11.74

#Univariate analysis with AST_home

lm.fit<-lm(PTS_home~AST_home, data=dfTraining)
summary(lm.fit)
actuals<-dfValidation$PTS_home
test_tmp<-data.frame(AST_home=dfValidation$AST_home)
lm.pred <- predict(lm.fit,test_tmp)
sqrt(mean((lm.pred-actuals)^2,na.rm = TRUE))  #RMSE = 10.4955

#Univariate analysis of REB_home

lm.fit<-lm(PTS_home~REB_home, data=dfTraining)
summary(lm.fit)
actuals<-dfValidation$PTS_home
test_tmp<-data.frame(REB_home=dfValidation$REB_home)
lm.pred <- predict(lm.fit,test_tmp)
sqrt(mean((lm.pred-actuals)^2,na.rm = TRUE))  #RMSE = 12.79566


#For applying the best model to test:

train_valid <- rbind(na.omit(dfTraining),na.omit(dfValidation))


train_data = model.matrix(PTS_home~., data = train_valid)
test_data = model.matrix(PTS_home~., data = na.omit(dfTest))

#Creating vectors for Y, getting rid of NA values 
y_train <- na.omit(train_valid$PTS_home)
y_test <- na.omit(dfTest$PTS_home)

#setting up general lasso model 
lasso_train = glmnet(train_data, y_train, alpha = 1, lambda = grid, thresh = 1e-12)

#using cross validation to identify the best lambda
crossValLasso_train = cv.glmnet(train_data, y_train, alpha = 1, lambda = grid, thresh = 1e-12)

#getting best lambda value using cross validation = 0.01
bestLasso_train = crossValLasso_train$lambda.min
#obtaining lambda 1 se away = 0.376
L_Lambda <- crossValLasso_train$lambda.1se

#lambda values vs RMSE to identify lambda
plot(log(crossValLasso_train$lambda),sqrt(crossValLasso_train$cvm),
     main="LASSO CV (k=10)",xlab="log(lambda)",
     ylab = "RMSE",col=4,type="b",cex.lab=1.2)
abline(v=log(L_Lambda),lty=2,col=2,lwd=2)
abline(v=log(bestLasso_train),lty=2,col=2,lwd=2)

#looking at coefficients for training lasso w best lambda
lasso_train1 = glmnet(train_data, y_train, alpha = 1, lambda = bestLasso_train, thresh = 1e-12)
coef(lasso_train1)


'''
                      s0
(Intercept)  -10.7023910
(Intercept)    .        
FG_PCT_home  120.5516755
FT_PCT_home   22.5578553
FG3_PCT_home  13.5511079
AST_home       0.5975036
REB_home       0.5105101
'''

#using best lambda to apply our lasso regression to our test data set 
lasso.pred_train = predict(crossValLasso_train, s = bestLasso_train, newx = test_data)
#compare with using L_Lambda 
lasso.pred_train1 = predict(crossValLasso_train, s = L_Lambda, newx = test_data)

#our model produced an RMSE of 
sqrt(mean((lasso.pred_train - y_test)^2)) #7.831
#compare with using L_Lambda 
sqrt(mean((lasso.pred_train1 - y_test)^2)) #7.877
#so choose lambda of 0.01 bc lower test error

#Coefficients Lasso regression model selected: 
Future2 = predict(crossValLasso_train, REB_home, interval = "prediction",se.fit=T,level=0.99)
coef.L_test = predict(crossValLasso_train,type="coefficients",s=bestLasso_train)
'''
(Intercept)  -10.7024097
(Intercept)    .        
FG_PCT_home  120.5517188
FT_PCT_home   22.5578562
FG3_PCT_home  13.5511097
AST_home       0.5975033
REB_home       0.5105102
'''
#compare with using L_Lambda 
coef.L_test1 = predict(crossValLasso_train,type="coefficients",s=L_Lambda)

'''
  1
(Intercept)   -1.239075
(Intercept)    .       
FG_PCT_home  115.238524
FT_PCT_home   18.769589
FG3_PCT_home  11.468141
AST_home       0.580527
REB_home       0.440378
'''


#calculating R^2
rss <- sum((lasso.pred_train - y_test) ^ 2)
tss <- sum((y_test - mean(y_test)) ^ 2)
rsq <- 1 - rss/tss
rsq #0.634

#train_valid_n = 17320
N = 17320
#K is number of independt variables = 5
k=5
adjusted_rsq = (1-(((1-rsq)*(N-1)) / (N-k-1))) #0.633

#plotting actual vs fitted
plot(lasso.pred_train, y_test, xlab="Predicted Y Values",ylab="Actual Y Values", main="Actual vs Fitted")

#plotting residuals
res = lasso.pred_train - y_test
plot(lasso.pred_train, res,
     pch=19, #Type of point
     xlab="Predicted Y Values",ylab="Residuals")
abline(0,0)
#observe a cluster

#QQ plot
qqnorm(res)
#add a straight diagonal line to the plot
qqline(res) 
#residuals stray from plot at tails.. not normally distributed?

#density plot 
plot(density(res), main="Distribution of Residuals")
#skewed left 

