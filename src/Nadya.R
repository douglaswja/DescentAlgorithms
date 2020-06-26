setwd("C:/UNI/DSA3102")
library(tidyverse)
library(tictoc)
Xtrain <- as.matrix(read.csv("Xtrain.csv", header=FALSE))
Xtest <- as.matrix(read.csv("Xtest.csv", header=FALSE))
ytrain <- as.matrix(read.csv("ytrain.csv", header=FALSE))
ytest <- as.matrix(read.csv("ytest.csv", header=FALSE))



armijo <- function(fun, fw, w, d, deltafw, alpha0, beta, omega, maxit){
  dot <- sum(d*deltafw)
  f_w_plus_td <- fun(w+alpha0*d)[[1]]
  rhs <- fw + alpha0*omega*dot
  i <- 0
  alpha <- alpha0
  while (f_w_plus_td > rhs){
    i <- i+1
    alpha <- alpha0*(beta^i)
    f_w_plus_td <- fun(w+alpha*d)[[1]]
    rhs <- fw + alpha*omega*dot
    if (i==maxit){
      cat("max iter reached")
      break
    }
  }
  return(list(alpha, i))
  
}


LL <- function(W){ #W is a column vector with size = numfeatures
  fx <- 0
  num_data <- dim(Xtrain)[1]
  num_features <- dim(Xtrain)[2]
  grad <- as.matrix(rep(0,num_features), nrow=num_features)
  for (i in 1:num_data){
    xi <- as.matrix(Xtrain[i,], nrow=num_features)
    yi <- ytrain[[i]]
    dum <- crossprod(yi*W, xi)
    fx <- fx + log(1+exp(-dum))
    dums <- (exp(dum)+1)[[1]]
    grad <- grad + (-yi*xi)/dums
  }
  return(list(fx, grad))
}


steepest_GD <- function(fun, w0, tol, maxit){
  w <- w0
  tol2 <- tol/100
  tbl <- data.frame(iter=as.integer(),normgrad=as.numeric(), f=as.numeric())
  tbl <- cbind(tbl, data.frame(matrix(nrow=0, ncol=57)))
  all_fw <- c()
  for (iter in 1:maxit){
    if (iter%%1000 == 0){
      cat(iter)
    }
    result <- fun(w)
    fw <- result[[1]]
    all_fw <- c(all_fw, fw)
    grad <- result[[2]]
    d <- (-1)*grad
    normd <- norm(d, type="2")
    if (iter==1){
      tbl[1,] <- c(iter, normd, fw, w)
    }
    if (normd < tol){
      tbl[2,] <- c(iter, normd, fw, w)
      break
    }
    armijo_res <- armijo(fun, fw, w, d, grad, 0.0001, 0.7, 0.3, 100)
    t <- armijo_res[[1]]
    w <- w + t*d
    
  }
  all_fw_df <- data.frame(iter=1:length(all_fw), lw=all_fw)  
  return(list(w,d,iter,tbl,all_fw_df))
  
}
tri <- steepest_GD(LL, as.matrix(rep(0.05,57),nrow=57), 0.1, 100)
#tol = 0.1, stored in table, wrong armijo

tic()
trial_GD_01 <- steepest_GD(LL, as.matrix(rep(0.05,57),nrow=57), 0.1, 50000)
time_GD_01 <- toc()
exec_time_GD_01 <- time_GD_01$toc - time_GD_01$tic #19093.57 sec
table_GD_01<- trial_GD_01[[4]] #number of iterations = 44423
ggplot(data=table_GD_01%>%filter(iter%%20==0), aes(x=iter,y=f)) +
  geom_point() +
  labs(title="Trial GD, tol = 0.1")
weights_GD_01 <- trial_GD_01[[1]]

best_w_GD_01 <- t(as.matrix(weights_GD_01))
y_pred_train_GD_01 <- apply(Xtrain,1,function(x){1/(1+exp((-1)*(best_w_GD_01%*%as.matrix(x,nrow=57))))})
y_pred_train_final_GD_01 <- sapply(y_pred_train_GD_01, function(x){ifelse(x>0.5,1,-1)})
accuracy_train_GD_01 <- sum(y_pred_train_final_GD_01==ytrain)/3065*100 #94.77977%
compare_GD_01 <- data.frame(ytrain,y_pred_train_final_GD_01)

y_pred_test_GD_01 <- apply(Xtest,1,function(x){1/(1+exp((-1)*(best_w_GD_01%*%as.matrix(x,nrow=57))))})
y_pred_test_final_GD_01 <- sapply(y_pred_test_GD_01, function(x){ifelse(x>0.5,1,-1)})
accuracy_test_GD_01 <- sum(y_pred_test_final_GD_01==ytest)/1536*100 #94.0755%

#tol = 0.01, not stored in table, wrong armijo

tic()
trial_GD_001 <- steepest_GD(LL, as.matrix(rep(0.05,57),nrow=57), 0.01, 2000000)
time_GD_001 <- toc()
exec_time_GD_001 <- time_GD_001$toc - time_GD_001$tic #19367.96 sec
table_GD_001<- trial_GD_001[[4]] #number of iterations = 104162
ggplot(data=table_GD_001%>%filter(iter%%20==0), aes(x=iter,y=f)) +
  geom_point() +
  labs(title="Trial GD, tol = 0.01")
weights_GD_001 <- trial_GD_001[[1]]

best_w_GD_001 <- t(as.matrix(weights_GD_001))
y_pred_train_GD_001 <- apply(Xtrain,1,function(x){1/(1+exp((-1)*(best_w_GD_001%*%as.matrix(x,nrow=57))))})
y_pred_train_final_GD_001 <- sapply(y_pred_train_GD_001, function(x){ifelse(x>0.5,1,-1)})
accuracy_train_GD_001 <- sum(y_pred_train_final_GD_001==ytrain)/3065*100 #94.77977%
compare_GD_001 <- data.frame(ytrain,y_pred_train_final_GD_001)

y_pred_test_GD_001 <- apply(Xtest,1,function(x){1/(1+exp((-1)*(best_w_GD_001%*%as.matrix(x,nrow=57))))})
y_pred_test_final_GD_001 <- sapply(y_pred_test_GD_001, function(x){ifelse(x>0.5,1,-1)})
accuracy_test_GD_001 <- sum(y_pred_test_final_GD_001==ytest)/1536*100 #94.0755%


#tol = 0.001, not stored in table, right armijo

tic()
trial_GD_0001 <- steepest_GD(LL, as.matrix(rep(0.05,57),nrow=57), 0.001, 200000000)
time_GD_0001 <- toc()
exec_time_GD_0001 <- time_GD_0001$toc - time_GD_0001$tic #     sec
table_GD_0001<- trial_GD_0001[[4]] #number of iterations = 
ggplot(data=table_GD_0001%>%filter(iter%%20==0), aes(x=iter,y=f)) +
  geom_point() +
  labs(title="Trial GD, tol = 0.001")
weights_GD_0001 <- trial_GD_0001[[1]]

best_w_GD_001 <- t(as.matrix(weights_GD_001))
y_pred_train_GD_001 <- apply(Xtrain,1,function(x){1/(1+exp((-1)*(best_w_GD_001%*%as.matrix(x,nrow=57))))})
y_pred_train_final_GD_001 <- sapply(y_pred_train_GD_001, function(x){ifelse(x>0.5,1,-1)})
accuracy_train_GD_001 <- sum(y_pred_train_final_GD_001==ytrain)/3065*100 #94.77977%
compare_GD_001 <- data.frame(ytrain,y_pred_train_final_GD_001)

y_pred_test_GD_001 <- apply(Xtest,1,function(x){1/(1+exp((-1)*(best_w_GD_001%*%as.matrix(x,nrow=57))))})
y_pred_test_final_GD_001 <- sapply(y_pred_test_GD_001, function(x){ifelse(x>0.5,1,-1)})
accuracy_test_GD_001 <- sum(y_pred_test_final_GD_001==ytest)/1536*100 #94.0755%
