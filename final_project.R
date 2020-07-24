rm(list=ls()) #Removes every object from your environment

#Read data
setwd("C:/Users/india/Documents/MSBASummer/predictive_analytics")
df <- read.csv("games.csv")
df1 <- df[,-c(14:20)] #drop the away team stats
df <- df1[,-c(2,5,7)] #drop game id, visitor team id, home id (duplicate column)

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

# Finally, output the three dataframes for training, validation and test.
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
