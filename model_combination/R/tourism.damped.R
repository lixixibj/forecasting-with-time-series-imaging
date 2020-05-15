library(Tcomp)
library(forecast)
mape_cal <- function(outsample, forecasts) {
  #Used to estimate sMAPE
  outsample <- as.numeric(outsample) ; forecasts<-as.numeric(forecasts)
  mape <- (abs(outsample-forecasts))/(abs(outsample))
  return(mape)
}


mase_cal <- function(insample, outsample, forecasts) {
  stopifnot(stats::is.ts(insample))
  #Used to estimate MASE
  frq <- stats::frequency(insample)
  forecastsNaiveSD <- rep(NA,frq)
  for (j in (frq+1):length(insample)){
    forecastsNaiveSD <- c(forecastsNaiveSD, insample[j-frq])
  }
  masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
  
  outsample <- as.numeric(outsample) ; forecasts <- as.numeric(forecasts)
  mase <- (abs(outsample-forecasts))/masep
  return(mase)
}
mape.vector=c()
mase.vector=c()
for (i in 1:length(tourism)) {
  fc2 <- holt(tourism[[i]]$x, damped=TRUE,h=tourism[[i]]$h)
  mape.vector=c(mape.vector,mean(mape_cal(tourism[[i]]$xx, fc2$mean)))
  mase.vector=c(mase.vector,mean(mase_cal(tourism[[i]]$x, tourism[[i]]$xx, fc2$mean)))
}

#total
mean(mape.vector)
mean(mase.vector)
#monthly 1:366
mean(mape.vector[1:366])
mean(mase.vector[1:366])
#quarterly 367:793
mean(mape.vector[367:793])
mean(mase.vector[367:793])
#yearly 794:1311
mean(mape.vector[794:1311])
mean(mase.vector[794:1311])