training_tmp<-na.omit(dfTraining)
library(gbm)
boostfit = gbm(PTS_home~., #Formula (. means that all variables of df are included)
               
               data=training_tmp, #Data
               distribution='gaussian',
               interaction.depth=2, #Maximum depth of each tree
               n.trees=500, #Number of trees
               shrinkage=.2) #Learning rate

Y_boost_validation<-dfValidation$PTS_home
boost.pred<-predict(boostfit, newdata = dfTest_2)
res_boost.test<-data.frame((Y_boost_validation - boost.pred)^2)
names(res_boost.test)[1] <- "Errors"
sqrt(mean(res_boost.test[, 'Errors'], na.rm = TRUE))

vsum=summary(boostfit,plotit=FALSE) #This will have the variable importance info



plot(vsum$rel.inf,ylab="Relative Importance",axes=F,pch=16,col='red')
axis(1,labels=vsum$var,at=1:5)
axis(2)
for(i in 1:5){
  lines(c(i,i),c(0,vsum$rel.inf[i]),lwd=4,col='blue')
}