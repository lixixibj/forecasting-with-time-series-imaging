library(Tcomp)
library(dplyr)
library(tidyr)
library(parallel)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('process_dataset.R')
# this function runs the four standard models in forecast_comp
# on a large chunk of the competition series from either Mcomp or Tcomp.  
# The aim is to help comparisons with Athanasopoulos et al.  
#
# The use of makePSOCKcluster and parLapply speeds up the analysis nearly four fold on my laptop
# eg running the test on all the yearly tourism series takes 12 seconds rather than 44 seconds.

#' @param dataobj a list of class Mcomp such as M3 or tourism
#' @param cond1 a condition for subsetting dataobj eg "yearly"
#' @param tests a list of different horizons at which to return the MASE for four different models
#' 
#' @return a data.frame with \code{length(tests) + 2} columns and 8 rows
accuracy_measures <- function(dataobj, cond1, tests){
  cores <- detectCores()
  
  cluster <- makePSOCKcluster(max(1, cores - 1))
  
  clusterEvalQ(cluster, {
    library(Tcomp)
    library(forecast)
  })
  
  results <- parLapply(cluster,
                       subset(dataobj, cond1), 
                       forecast_comp, 
                       tests = tests)
  
  results_mat <- do.call(rbind, results)
  nr <- nrow(results_mat)
  
  tmp <- as.data.frame(results_mat) %>%
    mutate(measure = rep(rep(c("MAPE", "MASE"), times = c(4, 4)), times = nr / 8)) %>%
    mutate(method = rownames(results_mat)) %>%
    gather(horizon, mase, -method, -measure) %>%
    group_by(method, measure, horizon) %>%
    summarise(result = round(mean(mase), 3)) %>%
    ungroup() %>%
    mutate(horizon = factor(horizon, levels = colnames(results[[1]]))) %>%
    spread(horizon, result) %>%
    arrange(measure) %>%
    as.data.frame()
  
  stopCluster(cluster)
  
  return(tmp)
}

accuracy_measures(tourism, "monthly", list(1, 2, 3, 6, 12, 18, 24, 1:3, 1:12, 1:24))
accuracy_measures(tourism, "quarterly", list(1, 2, 3, 4, 6, 8, 1:4, 1:8))
accuracy_measures(tourism, "yearly", list(1, 2, 3, 4, 1:2, 1:4))


#damped method
library(Tcomp)
library(forecast)
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

