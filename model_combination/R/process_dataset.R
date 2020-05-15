##########################################################
## Given a dataset with the correct format              ##
## forecast are applied to all series in the dataset    ##
## given their respective forecast horizons             ##
## and a new dataset is created with based on the input ##
## storing the individual forecasts, forecast errors,   ##
## and which method is best                             ##
##########################################################


# The input dataset is in the following format (names borrowed from the Mcomp package)
# a list with elements of the following structure
#  x   : The series
#  h   : The number of time steps required to forecast
#  xx  : The true future series (of length h)

# The output dataset will have the input structure, additionally,
# new entries in the structure will be added
# ff              :  A matrix with F rows and h columns, with F being the number of
#                   forecast methods.
# errors          : a vector of F elements containing the errors of the methods





#####################################################################
####### ERROR CALCULATION, TAKEN FROM THE M4 Competition GitHub  ####
####### https://github.com/M4Competition/M4-methods #################
#####################################################################


smape_cal <- function(outsample, forecasts) {
  #Used to estimate sMAPE
  outsample <- as.numeric(outsample) ; forecasts<-as.numeric(forecasts)
  smape <- (abs(outsample-forecasts)*200)/(abs(outsample)+abs(forecasts))
  return(smape)
}


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

SeasonalityTest <- function(input, ppy){
  #Used to determine whether a time series is seasonal
  tcrit <- 1.645
  if (length(input)<3*ppy){
    test_seasonal <- FALSE
  }else{
    xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]
    clim <- tcrit/sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
    test_seasonal <- ( abs(xacf[ppy]) > clim[ppy] )
    
    if (is.na(test_seasonal)==TRUE){ test_seasonal <- FALSE }
  }
  
  return(test_seasonal)
}
######caculate naive2##################
Benchmarks <- function(input, fh){
  #Used to estimate the statistical benchmarks of the M4 competition
  
  #Estimate seasonaly adjusted time series
  
  ppy <- frequency(input) ; ST <- F
  
  if (ppy>1){ ST <- SeasonalityTest(input,ppy)  }
  if (ST==T){
    Dec <- decompose(input,type="multiplicative")
    des_input <- input/Dec$seasonal
    SIout <- head(rep(Dec$seasonal[(length(Dec$seasonal)-ppy+1):length(Dec$seasonal)], fh), fh)
  }else{
    des_input <- input ; SIout <- rep(1, fh)
  }
  
  f3 <- naive(des_input, h=fh)$mean*SIout #Naive2
  
  return(f3)
}

calc_errors_new <- function(dataset) {
  
  total_naive2_errors <- c(0,0)
  for (i in 1:length(dataset)) {
    tryCatch({
      print(i)
      lentry <- dataset[[i]]
      insample <- lentry$x
      
      #extrac forecasts and attach the snaive for completion
      #ff <- lentry$ff
      y_hat<-lentry$ff
      #??ff??Ԥ??ֵ???·???
      #???߽?sniave ?ĳ?niave2??Ԥ?ⷽ???Ϳ?????
      y_hat <- rbind(y_hat, Benchmarks(insample, lentry$h))
      
      frq <- frq <- stats::frequency(insample)
      insample <- as.numeric(insample)
      outsample <- as.numeric(lentry$xx)
      #ǰh??ֵ?ͺ?h??ֵ
      masep <- mean(abs(utils::head(insample,-frq) - utils::tail(insample,-frq)))
      
      
      repoutsample <- matrix(
        rep(outsample, each=nrow(y_hat)),
        nrow=nrow(y_hat))
      
      smape_err <- 200*abs(y_hat - repoutsample) / (abs(y_hat) + abs(repoutsample))
      
      mase_err <- abs(y_hat - repoutsample) / masep
      
      lentry$naive2_mase <- mase_err[nrow(mase_err), ]
      lentry$naive2_smape <- smape_err[nrow(smape_err),]
      
      lentry$mase_err <- mase_err[-nrow(mase_err),]
      lentry$smape_err <- smape_err[-nrow(smape_err),]
      dataset[[i]] <- lentry
      total_naive2_errors <- total_naive2_errors + c(mean(lentry$naive2_mase),
                                                     mean(lentry$naive2_smape))
    } , error = function (e) {
      print(paste("Error when processing OWIs in series: ", i))
      print(e)
      e
    })
  }
  total_naive2_errors = total_naive2_errors / length(dataset)
  avg_naive2_errors <- list(avg_mase=total_naive2_errors[1],
                            avg_smape=total_naive2_errors[2])
  
  
  for (i in 1:length(dataset)) {
    lentry <- dataset[[i]]
    dataset[[i]]$errors <- 0.5*(rowMeans(lentry$mase_err)/avg_naive2_errors$avg_mase +
                                  rowMeans(lentry$smape_err)/avg_naive2_errors$avg_smape)
    #print(i)
    #dataset[[i]]$errors <- rowMeans(lentry$smape_err)
  }
  attr(dataset, "avg_naive2_errors") <- avg_naive2_errors
  dataset
}

calc_errors_new_for_mape <- function(dataset) {
  
  total_naive2_errors <- c(0,0)
  for (i in 1:length(dataset)) {
    tryCatch({
      print(i)
      lentry <- dataset[[i]]
      insample <- lentry$x
      
      #extrac forecasts and attach the snaive for completion
      #ff <- lentry$ff
      y_hat<-lentry$ff
      #??ff??Ԥ??ֵ???·???
      #???߽?sniave ?ĳ?niave2??Ԥ?ⷽ???Ϳ?????
      y_hat <- rbind(y_hat, Benchmarks(insample, lentry$h))
      
      frq <- frq <- stats::frequency(insample)
      insample <- as.numeric(insample)
      outsample <- as.numeric(lentry$xx)
      #ǰh??ֵ?ͺ?h??ֵ
      masep <- mean(abs(utils::head(insample,-frq) - utils::tail(insample,-frq)))
      
      
      repoutsample <- matrix(
        rep(outsample, each=nrow(y_hat)),
        nrow=nrow(y_hat))
      #mape 20200515
      smape_err <- abs(y_hat - repoutsample) / (abs(repoutsample))
      
      mase_err <- abs(y_hat - repoutsample) / masep
      
      lentry$naive2_mase <- mase_err[nrow(mase_err), ]
      lentry$naive2_smape <- smape_err[nrow(smape_err),]
      
      lentry$mase_err <- mase_err[-nrow(mase_err),]
      lentry$smape_err <- smape_err[-nrow(smape_err),]
      dataset[[i]] <- lentry
      total_naive2_errors <- total_naive2_errors + c(mean(lentry$naive2_mase),
                                                     mean(lentry$naive2_smape))
    } , error = function (e) {
      print(paste("Error when processing OWIs in series: ", i))
      print(e)
      e
    })
  }
  total_naive2_errors = total_naive2_errors / length(dataset)
  avg_naive2_errors <- list(avg_mase=total_naive2_errors[1],
                            avg_smape=total_naive2_errors[2])
  
  
  for (i in 1:length(dataset)) {
    lentry <- dataset[[i]]
    dataset[[i]]$errors <- 0.5*(rowMeans(lentry$mase_err)/avg_naive2_errors$avg_mase +
                                  rowMeans(lentry$smape_err)/avg_naive2_errors$avg_smape)
    #print(i)
    #dataset[[i]]$errors <- rowMeans(lentry$smape_err)
  }
  attr(dataset, "avg_naive2_errors") <- avg_naive2_errors
  dataset
}


############################################################################
####### CALULATE THE FORECASTS AND ERRORS FOR A GIVEN FORECAST METHOD ######
############################################################################

# Given a list of methods, it is applied to the element in the input dataset
# and the element of the output dataset is generated
# the list of R functions is used for easy application of many methods
# the name of the methods in the output entry forec.methods is taken from the functions
# as strings.
# These R functions should take as input a parameter x (the time series) and
# h (the forecast horizon)
# and output only a vector of h elements with the forecasts.
# This way any method in the forecast package can be easily added to the list, and also other
#custom methods



#calculate forecast predictions
calculate_forecast_preds <- function(insample, h, forec.method) {
  forec.method(x=insample, h=h)
}


#calc SMAPE and MASE errors for a given model and parameters
calculate_errors <- function(insample, outsample, forecasts) {
  SMAPE <- smape_cal(outsample, forecasts)
  MASE <- mase_cal(insample, outsample, forecasts)
  c(mean(SMAPE) , mean(MASE))
}

#output only the owi error
calculate_owi <- function(insample, outsample, snaive_errors, forecasts) {
  errors <- calculate_errors(insample, outsample, forecasts)
  0.5*( (errors[1] / snaive_errors[1]) +  (errors[2]/snaive_errors[2]))
}


#processes forecast methods on a series
#given a series component
#and list of forecast methods
process_forecast_methods <- function(seriesdata, methods_list) {

  #process each method in methods_list to produce the forecasts and the errors
  lapply(methods_list, function (mentry) {
    method_name <- mentry
    method_fun <- get(mentry)
    forecasts <- tryCatch( method_fun(x=seriesdata$x, h=seriesdata$h),
                           error=function(error) {
                             print(error)
                             print(paste("ERROR processing series: ", seriesdata$st))
                             print(paste("The forecast method that produced the error is:",
                                         method_name))
                             print("Returning snaive forecasts instead")
                             snaive_forec(seriesdata$x, seriesdata$h)
                           })
    list( forecasts=forecasts, method_name=method_name)
  })
}


#' Generate Forecasts for a Time Series Dataset
#'
#' For each series in \code{dataset}, forecasts
#' are generated for all methods in \code{methods}.
#'
#' \code{dataset} must be a list with each element having the following format:
#' \describe{
#'   \item{x}{A time series object \code{ts} with the historical data.}
#'   \item{h}{The number of required forecasts.}
#' }
#'
#' \code{methods} is a list of strings with the names of the functions that generate the
#' forecasts. The functions must exist and take as parameters (\code{x}, \code{h}), with
#' \code{x} being the \code{ts} object with the input series and \code{h} the number of required
#' forecasts (after the last observation of \code{x}). The output of these functions must be
#' a vector or \code{ts} object of length \code{h} with the produced forecast.
#' No additional parameters are required in the functions.
#'
#' @param dataset The list containing the series. See details for the required format.
#' @param methods A list of strings with the names of the functions that generate
#' the forecasts.
#' @param n.cores The number of cores to be used. \code{n.cores > 1} means parallel processing.
#'
#' @return A list with the elements having the following structure
#' \describe{
#'   \item{x}{A time series object \code{ts} with the historical data.}
#'   \item{h}{The number of required forecasts.}

#'   \item{ff}{A matrix with F rows and \code{h} columns. Each row contains
#'   the forecasts of each method in \code{methods} }
#' }
#'
#' @examples
#' auto_arima_forec <- function(x, h) {
#'   model <- forecast::auto.arima(x, stepwise=FALSE, approximation=FALSE)
#'   forecast::forecast(model, h=h)$mean
#' }
#'
#' snaive_forec <- function(x,h) {
#'   model <- forecast::snaive(x, h=length(x))
#'   forecast::forecast(model, h=h)$mean
#' }
#' rw_drift_forec <- function(x, h) {
#'   model <- forecast::rwf(x, drift=TRUE, h=length(x))
#'  forecast::forecast(model, h=h)$mean
#' }
#'
#' create_example_list <- function() {
#'   methods <- list("auto_arima_forec")
#'   methods <- append(methods, "snaive_forec")
#'   methods <- append(methods, "rw_drift_forec")
#'   methods
#' }
#' methods <- create_example_list()
#' forec_results <- calc_forecasts(Mcomp::M3[1:4], methods, n.cores=1)
#'
#' @export
calc_forecasts <- function(dataset, methods, n.cores=1) {
  list_process_fun <- lapply
  cl = -1

  if (n.cores > 1) {
    cl <- parallel::makeCluster(n.cores)
    .env <- as.environment(as.list(environment(process_forecast_methods), all.names = TRUE))
    lapply(methods, function(method) assign(method, get(method), .env))
    parallel::clusterExport(cl, varlist=ls(envir=.env), envir = .env)
    list_process_fun <- function(my_list, ...) {
      parallel::parLapplyLB(cl, my_list, ...)
    }
  }

  ret_list <- list_process_fun(dataset, function (seriesdata) {
    results <- process_forecast_methods(seriesdata, methods)
    ff <- t(sapply(results, function (resentry) resentry$forecasts))
    method_names <- sapply(results, function (resentry) resentry$method_name)
    row.names(ff) <- method_names
    seriesdata$ff <- ff
    seriesdata
  })

  if (n.cores > 1) {
    parallel::stopCluster(cl)
  }

  ret_list
}

