rm(list=ls())

df <- read.csv("games.csv")
df1 <- df[,-c(14:20)] 
df <- df1[,-c(2,5,7)]

#sample size
n = dim(df)[1]
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

# Finally, output the three dataframes for training, validation and test.
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
training_tmp<-na.omit(dfTraining)
boostfit = gbm(PTS_home~., #Formula (. means that all variables of df are included)
               
               data=training_tmp, #Data
               distribution='gaussian',
               interaction.depth=2, #Maximum depth of each tree
               n.trees=100, #Number of trees
               shrinkage=.2) #Learning rate

par(mfrow=c(1,1)) #Plot window: 1 row, 1 column
p=ncol(Boston)-1 #Number of covariates (-1 because one column is the response)
vsum=summary(boostfit,plotit=FALSE) #This will have the variable importance info
row.names(vsum)=NULL #Drop variable names from rows.

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


############# Multivariate Analysis ####################################

lm.fit<-lm(PTS_home~., data=dfTraining)
summary(lm.fit)
actuals<-dfTest$PTS_home
test_tmp<-dfTest[,-c(1:5,11)]
lm.pred <- predict(lm.fit,test_tmp)
sqrt(mean((lm.pred-actuals)^2,na.rm = TRUE))  #RMSE = 7.83













