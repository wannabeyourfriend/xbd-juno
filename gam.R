#!/usr/bin/env Rscript
## 使用 GAM 进行拟合

library(arrow)
library(argparser)
library(mgcv)
library(proto)
psr <- arg_parser("GAM fit.")
psr <- add_argument(psr, "--ipt", nargs=Inf, help="parquet input of histogram")
psr <- add_argument(psr, "--b", help="number of spatial bins", type="integer")
psr <- add_argument(psr, "--t", help="number of timing bins", type="integer")
psr <- add_argument(psr, "--opt", help="RDS object of GAM model")
args <- parse_args(psr)

t_max <- 1000
t_binwidth <- t_max / args$t

# 从18行-520行，这段代码是为了解除数据长度的限制，请同学们不要做任何改动！！！
my.smoothCon <- function(object,data,knots=NULL,absorb.cons=FALSE,scale.penalty=TRUE,n=nrow(data),
                      dataX = NULL,null.space.penalty = FALSE,sparse.cons=0,diagonal.penalty=FALSE,
                      apply.by=TRUE,modCon=0)
## wrapper function which calls smooth.construct methods, but can then modify
## the parameterization used. If absorb.cons==TRUE then a constraint free
## parameterization is used. 
## Handles `by' variables, and summation convention.
## If a smooth has an entry 'sumConv' and it is set to FALSE, then the summation convention is
## not applied to matrix arguments. 
## apply.by==FALSE causes by variable handling to proceed as for apply.by==TRUE except that
## a copy of the model matrix X0 is stored for which the by variable (or dummy) is never
## actually multiplied into the model matrix. This facilitates
## discretized fitting setup, where such multiplication needs to be handled `on-the-fly'.
## Note that `data' must be a data.frame or model.frame, unless n is provided explicitly, 
## in which case a list will do.
## If present dataX specifies the data to be used to set up the model matrix, given the 
## basis set up using data (but n same for both).
## modCon: 0 (do nothing); 1 (delete supplied con); 2 (set fit and predict to predict)
##         3 (set fit and predict to fit)
{ sm <- smooth.construct3(object,data,knots)
  if (!is.null(attr(sm,"qrc"))) warning("smooth objects should not have a qrc attribute.")
  if (modCon==1) sm$C <- sm$Cp <- NULL ## drop any supplied constraints in favour of auto-cons
  ## add plotting indicator if not present.
  ## plot.me tells `plot.gam' whether or not to plot the term
  if (is.null(sm$plot.me)) sm$plot.me <- TRUE

  ## add side.constrain indicator if missing
  ## `side.constrain' tells gam.side, whether term should be constrained
  ## as a result of any nesting detected... 
  if (is.null(sm$side.constrain)) sm$side.constrain <- TRUE

  ## automatically produce centering constraint...
  ## must be done here on original model matrix to ensure same
  ## basis for all `id' linked terms...
  if (!is.null(sm$g.index)&&is.null(sm$C)) { ## then it's a monotonic smooth or a tensor product with monotonic margins
    ## compute the ingredients for sweep and drop cons...
    sm$C <- matrix(colMeans(sm$X),1,ncol(sm$X))
    if (length(sm$S)) {
      upen <- rowMeans(abs(sm$S[[1]]))==0 ## identify unpenalized
      if (length(sm$S)>1) for (i in 2:length(sm$S)) upen <- upen &  rowMeans(abs(sm$S[[i]]))==0
      if (sum(upen)>0) drop <- min(which(upen)) else {
        drop <- min(which(!sm$g.index))
      }
    } else drop <- which.min(apply(sm$X,2,sd))
    if (absorb.cons) sm$g.index <- sm$g.index[-drop] 
  } else drop <- -1 ## signals not to use sweep and drop (may be modified below)

  ## can this term be safely re-parameterized?
  if (is.null(sm$repara)) sm$repara <- if (is.null(sm$g.index)) TRUE else FALSE

  if (is.null(sm$C)) {
    if (sparse.cons<=0) {
      sm$C <- matrix(colMeans(sm$X),1,ncol(sm$X))
      ## following 2 lines implement sweep and drop constraints,
      ## which are computationally faster than QR null space
      ## however note that these are not appropriate for 
      ## models with by-variables requiring constraint! 
      if (sparse.cons == -1) { 
        vcol <- apply(sm$X,2,var) ## drop least variable column
        drop <- min((1:length(vcol))[vcol==min(vcol)])
      }
    } else if (sparse.cons>0) { ## use sparse constraints for sparse terms
      if (sum(sm$X==0)>.1*sum(sm$X!=0)) { ## treat term as sparse
        if (sparse.cons==1) {
          xsd <- apply(sm$X,2,FUN=sd)
          if (sum(xsd==0)) ## are any columns constant?
            sm$C <- ((1:length(xsd))[xsd==0])[1] ## index of coef to set to zero
          else {
            ## xz <- colSums(sm$X==0) 
            ## find number of zeroes per column (without big memory footprint)...
            xz <- apply(sm$X,2,FUN=function(x) {sum(x==0)}) 
            sm$C <- ((1:length(xz))[xz==min(xz)])[1] ## index of coef to set to zero
          }
        } else if (sparse.cons==2) {
            sm$C = -1 ## params sum to zero
        } else  { stop("unimplemented sparse constraint type requested") }
      } else { ## it's not sparse anyway 
        sm$C <- matrix(colSums(sm$X),1,ncol(sm$X))
      }
    } else { ## end of sparse constraint handling
      sm$C <- matrix(colSums(sm$X),1,ncol(sm$X)) ## default dense case
    }
    ## conSupplied <- FALSE
    alwaysCon <- FALSE
  } else { ## sm$C supplied
    if (modCon==2&&!is.null(sm$Cp)) sm$C <- sm$Cp ## reset fit con to predict
    if (modCon>=3) sm$Cp <- NULL ## get rid of separate predict con
    ## should supplied constraint be applied even if not needed? 
    if (is.null(attr(sm$C,"always.apply"))) alwaysCon <- FALSE else alwaysCon <- TRUE
  }

  ## set df fields (pre-constraint)...
  if (is.null(sm$df)) sm$df <- sm$bs.dim

  ## automatically discard penalties for fixed terms...
  if (!is.null(object$fixed)&&object$fixed) {
    sm$S <- NULL
  }

  ## The following is intended to make scaling `nice' for better gamm performance.
  ## Note that this takes place before any resetting of the model matrix, and 
  ## any `by' variable handling. From a `gamm' perspective this is not ideal, 
  ## but to do otherwise would mess up the meaning of smoothing parameters
  ## sufficiently that linking terms via `id's would not work properly (they 
  ## would have the same basis, but different penalties)

  sm$S.scale <- rep(1,length(sm$S))

  if (scale.penalty && length(sm$S)>0 && is.null(sm$no.rescale)) # then the penalty coefficient matrix is rescaled
  {  maXX <- norm(sm$X,type="I")^2 ##mean(abs(t(sm$X)%*%sm$X)) # `size' of X'X
      for (i in 1:length(sm$S)) {
        maS <- norm(sm$S[[i]])/maXX  ## mean(abs(sm$S[[i]])) / maXX
        sm$S[[i]] <- sm$S[[i]] / maS
        sm$S.scale[i] <- maS ## multiply S[[i]] by this to get original S[[i]]
      } 
  } 

  ## check whether different data to be used for basis setup
  ## and model matrix... 
  if (!is.null(dataX)) { er <- Predict.matrix3(sm,dataX) 
    sm$X <- er$X
    sm$ind <- er$ind
    rm(er)
  }

  ## check whether smooth called with matrix argument
  if ((is.null(sm$ind)&&nrow(sm$X)!=n)||(!is.null(sm$ind)&&length(sm$ind)!=n)) { 
    matrixArg <- TRUE 
    ## now get the number of columns in the matrix argument...
    if (is.null(sm$ind)) q <- nrow(sm$X)/n else q <- length(sm$ind)/n
    if (!is.null(sm$by.done)) warning("handling `by' variables in smooth constructors may not work with the summation convention ")
  } else {
    matrixArg <- FALSE
    if (!is.null(sm$ind)) {  ## unpack model matrix + any offset
      offs <- attr(sm$X,"offset")
      sm$X <- sm$X[sm$ind,,drop=FALSE]      
      if (!is.null(offs)) attr(sm$X,"offset") <- offs[sm$ind]
    }
  }
  offs <- NULL

  ## pick up "by variables" now, and handle summation convention ...

  if (matrixArg||(object$by!="NA"&&is.null(sm$by.done))) {
    #drop <- -1 ## sweep and drop constraints inappropriate
    if (is.null(dataX)) by <- get.var(object$by,data) 
    else by <- get.var(object$by,dataX)
    if (matrixArg&&is.null(by)) { ## then by to be taken as sequence of 1s
      if (is.null(sm$ind)) by <- rep(1,nrow(sm$X)) else by <- rep(1,length(sm$ind))
    }
    if (is.null(by)) stop("Can't find by variable")
    offs <- attr(sm$X,"offset")
    if (!is.factor(by)) {
     ## test for cases where no centring constraint on the smooth is needed. 
      if (!alwaysCon) {
        if (matrixArg) {
          L1 <- as.numeric(matrix(by,n,q)%*%rep(1,q))
          if (sd(L1)>mean(L1)*.Machine$double.eps*1000) { 
            ## sml[[1]]$C <- 
            sm$C <- matrix(0,0,1)
            ## if (!is.null(sm$Cp)) sml[[1]]$Cp <- sm$Cp <- NULL
            if (!is.null(sm$Cp)) sm$Cp <- NULL
          } else sm$meanL1 <- mean(L1) 
          ## else sml[[1]]$meanL1 <- mean(L1) ## store mean of L1 for use when adding intercept variability
        } else { ## numeric `by' -- constraint only needed if constant
          if (sd(by)>mean(by)*.Machine$double.eps*1000) { 
            ## sml[[1]]$C <- 
            sm$C <- matrix(0,0,1)   
            ## if (!is.null(sm$Cp)) sml[[1]]$Cp <- sm$Cp <- NULL
            if (!is.null(sm$Cp)) sm$Cp <- NULL
          }
        }
      } ## end of constraint removal
    }
  } ## end of initial setup of by variables

  if (absorb.cons&&drop>0&&nrow(sm$C)>0) { ## sweep and drop constraints have to be applied before by variables
     if (!is.null(sm$by.done)) warning("sweep and drop constraints unlikely to work well with self handling of by vars")
     qrc <- c(drop,as.numeric(sm$C)[-drop])
     class(qrc) <- "sweepDrop"
     sm$X <- sm$X[,-drop,drop=FALSE] - matrix(qrc[-1],nrow(sm$X),ncol(sm$X)-1,byrow=TRUE)
     if (length(sm$S)>0)
     for (l in 1:length(sm$S)) { # some smooths have > 1 penalty 
        sm$S[[l]]<-sm$S[[l]][-drop,-drop]
     }
     attr(sm,"qrc") <- qrc
     attr(sm,"nCons") <- 1
     sm$Cp <- sm$C <- 0  
     sm$rank <- pmin(sm$rank,ncol(sm$X))
     sm$df <- sm$df - 1
     sm$null.space.dim <- max(0,sm$null.space.dim-1)
  }

  if (matrixArg||(object$by!="NA"&&is.null(sm$by.done))) { ## apply by variables
    if (is.factor(by)) { ## generates smooth for each level of by
      if (matrixArg) stop("factor `by' variables can not be used with matrix arguments.")
      sml <- list()
      lev <- levels(by)
      ## if by variable is an ordered factor then first level is taken as a 
      ## reference level, and smooths are only generated for the other levels
      ## this can help to ensure identifiability in complex models. 
      if (is.ordered(by)&&length(lev)>1) lev <- lev[-1]
      #sm$rank[length(sm$S)+1] <- ncol(sm$X) ## TEST CENTERING PENALTY
      #sm$C <- matrix(0,0,1) ## TEST CENTERING PENALTY
      for (j in 1:length(lev)) {
        sml[[j]] <- sm  ## replicate smooth for each factor level
        by.dum <- as.numeric(lev[j]==by)
        sml[[j]]$X <- by.dum*sm$X   ## multiply model matrix by dummy for level

        #sml[[j]]$S[[length(sm$S)+1]] <- crossprod(sm$X[by.dum==1,]) ## TEST CENTERING PENALTY
	
        sml[[j]]$by.level <- lev[j] ## store level
        sml[[j]]$label <- paste(sm$label,":",object$by,lev[j],sep="") 
        if (!is.null(offs)) {
          attr(sml[[j]]$X,"offset") <- offs*by.dum
        }
      }
    } else { ## not a factor by variable
      sml <- list(sm)
      if ((is.null(sm$ind)&&length(by)!=nrow(sm$X))||
          (!is.null(sm$ind)&&length(by)!=length(sm$ind))) stop("`by' variable must be same dimension as smooth arguments")
     
      if (matrixArg) { ## arguments are matrices => summation convention used
        #if (!apply.by) warning("apply.by==FALSE unsupported in matrix case")
        if (is.null(sm$ind)) { ## then the sm$X is in unpacked form
          sml[[1]]$X <- as.numeric(by)*sm$X ## normal `by' handling
          ## Now do the summation stuff....
          ind <- 1:n 
          X <- sml[[1]]$X[ind,,drop=FALSE]
          for (i in 2:q) {
            ind <- ind + n
            X <- X + sml[[1]]$X[ind,,drop=FALSE]
          }
          sml[[1]]$X <- X
          if (!is.null(offs)) { ## deal with any term specific offset (i.e. sum it too)
            ## by variable multiplied version...
            offs <- attr(sm$X,"offset")*as.numeric(by)  
            ind <- 1:n 
            offX <- offs[ind,]
            for (i in 2:q) {
              ind <- ind + n
              offX <- offX + offs[ind,]
            }
            attr(sml[[1]]$X,"offset") <- offX
          } ## end of term specific offset handling
        } else { ## model sm$X is in packed form to save memory
          ind <- 0:(q-1)*n
          offs <- attr(sm$X,"offset")
          if (!is.null(offs)) offX <- rep(0,n) else offX <- NULL 
          sml[[1]]$X <- matrix(0,n,ncol(sm$X))  
          for (i in 1:n) { ## in this case have to work down the rows
            ind <- ind + 1
            sml[[1]]$X[i,] <- colSums(by[ind]*sm$X[sm$ind[ind],,drop=FALSE])
            if (!is.null(offs)) {
              offX[i] <- sum(offs[sm$ind[ind]]*by[ind])
            }      
          } ## finished all rows
          attr(sml[[1]]$X,"offset") <- offX
        } 
      } else {  ## arguments not matrices => not in packed form + no summation needed
        sml[[1]]$X <- as.numeric(by)*sm$X
        if (!is.null(offs)) attr(sml[[1]]$X,"offset") <- if (apply.by) offs*as.numeric(by) else offs
      }

      if (object$by == "NA") sml[[1]]$label <- sm$label else 
        sml[[1]]$label <- paste(sm$label,":",object$by,sep="") 
     
    } ## end of not factor by branch
  } else { ## no by variables
    sml <- list(sm)
  }

  ###########################
  ## absorb constraints.....#
  ###########################

  if (absorb.cons) {
    k<-ncol(sm$X)

    ## If Cp is present it denotes a constraint to use in place of the fitting constraints
    ## when predicting. 

    if (!is.null(sm$Cp)&&is.matrix(sm$Cp)) { ## identifiability cons different for prediction
      pj <- nrow(sm$Cp)
      qrcp <- qr(t(sm$Cp), LAPACK = TRUE) 
      for (i in 1:length(sml)) { ## loop through smooth list
        sml[[i]]$Xp <- t(qr.qty(qrcp,t(sml[[i]]$X))[(pj+1):k,]) ## form XZ
        sml[[i]]$Cp <- NULL 
        if (length(sml[[i]]$S)) { ## gam.side requires penalties in prediction para
          sml[[i]]$Sp <- sml[[i]]$S ## penalties in prediction parameterization
          for (l in 1:length(sml[[i]]$S)) { # some smooths have > 1 penalty 
            ZSZ <- qr.qty(qrcp,sml[[i]]$S[[l]])[(pj+1):k,]
            sml[[i]]$Sp[[l]]<-t(qr.qty(qrcp,t(ZSZ))[(pj+1):k,]) ## Z'SZ
          }
        }
      }
    } else qrcp <- NULL ## rest of Cp processing is after C processing

    if (is.matrix(sm$C)) { ## the fit constraints
      j <- nrow(sm$C)
      if (j>0) { # there are constraints
        indi <- (1:ncol(sm$C))[colSums(sm$C)!=0] ## index of non-zero columns in C
        nx <- length(indi)
        if (nx < ncol(sm$C)&&drop<0) { ## then some parameters are completely constraint free
          nc <- j ## number of constraints
          nz <- nx-nc   ## reduced null space dimension
          qrc <- qr(t(sm$C[,indi,drop=FALSE]), LAPACK = TRUE) ## gives constraint null space for constrained only
          for (i in 1:length(sml)) { ## loop through smooth list
            if (length(sm$S)>0)
            for (l in 1:length(sm$S)) # some smooths have > 1 penalty 
            { ZSZ <- sml[[i]]$S[[l]]
              if (nz>0) ZSZ[indi[1:nz],]<-qr.qty(qrc,sml[[i]]$S[[l]][indi,,drop=FALSE])[(nc+1):nx,] 
              ZSZ <- ZSZ[-indi[(nz+1):nx],]   
              if (nz>0) ZSZ[,indi[1:nz]]<-t(qr.qty(qrc,t(ZSZ[,indi,drop=FALSE]))[(nc+1):nx,])
              sml[[i]]$S[[l]] <- ZSZ[,-indi[(nz+1):nx],drop=FALSE]  ## Z'SZ

              ## ZSZ<-qr.qty(qrc,sm$S[[l]])[(j+1):k,]
              ## sml[[i]]$S[[l]]<-t(qr.qty(qrc,t(ZSZ))[(j+1):k,]) ## Z'SZ
            }
            if (nz>0) sml[[i]]$X[,indi[1:nz]]<-t(qr.qty(qrc,t(sml[[i]]$X[,indi,drop=FALSE]))[(nc+1):nx,])
            sml[[i]]$X <- sml[[i]]$X[,-indi[(nz+1):nx]]
            ## sml[[i]]$X<-t(qr.qty(qrc,t(sml[[i]]$X))[(j+1):k,]) ## form XZ
            attr(sml[[i]],"qrc") <- qrc
            attr(sml[[i]],"nCons") <- j;
            attr(sml[[i]],"indi") <- indi ## index of constrained parameters
            sml[[i]]$C <- NULL
            sml[[i]]$rank <- pmin(sm$rank,k-j)
            sml[[i]]$df <- sml[[i]]$df - j
            sml[[i]]$null.space.dim <- max(0,sml[[i]]$null.space.dim - j)
            ## ... so qr.qy(attr(sm,"qrc"),c(rep(0,nrow(sm$C)),b)) gives original para.'s
          } ## end smooth list loop
        } else { 
          { ## full QR based approach
            qrc<-qr(t(sm$C), LAPACK = TRUE) 
            for (i in 1:length(sml)) { ## loop through smooth list
              if (length(sm$S)>0)
              for (l in 1:length(sm$S)) { # some smooths have > 1 penalty 
                ZSZ<-qr.qty(qrc,sm$S[[l]])[(j+1):k,]
                sml[[i]]$S[[l]]<-t(qr.qty(qrc,t(ZSZ))[(j+1):k,]) ## Z'SZ
              }
              sml[[i]]$X <- t(qr.qty(qrc,t(sml[[i]]$X))[(j+1):k,]) ## form XZ
            }  
            ## ... so qr.qy(attr(sm,"qrc"),c(rep(0,nrow(sm$C)),b)) gives original para.'s
            ## and qr.qy(attr(sm,"qrc"),rbind(rep(0,length(b)),diag(length(b)))) gives 
            ## null space basis Z, such that Zb are the original params, subject to con. 
          }
          for (i in 1:length(sml)) { ## loop through smooth list
            attr(sml[[i]],"qrc") <- qrc
            attr(sml[[i]],"nCons") <- j;
            sml[[i]]$C <- NULL
            sml[[i]]$rank <- pmin(sm$rank,k-j)
            sml[[i]]$df <- sml[[i]]$df - j
            sml[[i]]$null.space.dim <- max(0,sml[[i]]$null.space.dim-j)
          } ## end smooth list loop
        } # end full null space version of constraint
      } else { ## no constraints
        for (i in 1:length(sml)) {
         attr(sml[[i]],"qrc") <- "no constraints"
         attr(sml[[i]],"nCons") <- 0;
        }
      } ## end else no constraints
    } else if (length(sm$C)>1) { ## Kronecker product of sum-to-zero contrasts (first element unused to allow index for alternatives)
      m <- sm$C[-1] ## contrast order
      for (i in 1:length(sml)) { ## loop through smooth list
        if (length(sm$S)>0)
        for (l in 1:length(sm$S)) { # some smooths have > 1 penalty 
          sml[[i]]$S[[l]] <- XZKr(XZKr(sml[[i]]$S[[l]],m),m)
        }
	p <- ncol(sml[[i]]$X) 
        sml[[i]]$X <- t(XZKr(sml[[i]]$X,m))
	total.null.dim <- prod(m-1)*p/prod(m)
	nc <- p - prod(m-1)*p/prod(m)
	attr(sml[[i]],"nCons") <- nc
        attr(sml[[i]],"qrc") <- c(sm$C,nc) ## unused, dim1, dim2, ..., n.cons
	sml[[i]]$C <- NULL
        ## NOTE: assumption here is that constructor returns rank, null.space.dim
	## and df, post constraint.
      }	
    } else if (sm$C>0) { ## set to zero constraints
       for (i in 1:length(sml)) { ## loop through smooth list
          if (length(sm$S)>0)
          for (l in 1:length(sm$S)) { # some smooths have > 1 penalty 
            sml[[i]]$S[[l]] <- sml[[i]]$S[[l]][-sm$C,-sm$C]
          }
          sml[[i]]$X <- sml[[i]]$X[,-sm$C]
          attr(sml[[i]],"qrc") <- sm$C
          attr(sml[[i]],"nCons") <- 1;
          sml[[i]]$C <- NULL
          sml[[i]]$rank <- pmin(sm$rank,k-1)
          sml[[i]]$df <- sml[[i]]$df - 1
          sml[[i]]$null.space.dim <- max(sml[[i]]$null.space.dim-1,0)
          ## so insert an extra 0 at position sm$C in coef vector to get original
        } ## end smooth list loop
    } else if (sm$C <0) { ## params sum to zero 
       for (i in 1:length(sml)) { ## loop through smooth list
          if (length(sm$S)>0)
          for (l in 1:length(sm$S)) { # some smooths have > 1 penalty 
            sml[[i]]$S[[l]] <- diff(t(diff(sml[[i]]$S[[l]])))
          }
          sml[[i]]$X <- t(diff(t(sml[[i]]$X)))
          attr(sml[[i]],"qrc") <- sm$C
          attr(sml[[i]],"nCons") <- 1;
          sml[[i]]$C <- NULL
          sml[[i]]$rank <- pmin(sm$rank,k-1)
          sml[[i]]$df <- sml[[i]]$df - 1
          sml[[i]]$null.space.dim <- max(sml[[i]]$null.space.dim-1,0)
          ## so insert an extra 0 at position sm$C in coef vector to get original
        } ## end smooth list loop       
    }
   
    ## finish off treatment of case where prediction constraints are different
    if (!is.null(qrcp)) {
      for (i in 1:length(sml)) { ## loop through smooth list
        attr(sml[[i]],"qrc") <- qrcp
        if (pj!=attr(sml[[i]],"nCons")) stop("Number of prediction and fit constraints must match")
        attr(sml[[i]],"indi") <- NULL ## no index of constrained parameters for Cp
      }
    }

  } else for (i in 1:length(sml)) attr(sml[[i]],"qrc") <-NULL ## no absorption

  ## now convert single penalties to identity matrices, if requested.
  ## This is relatively expensive, so is not routinely done. However
  ## for expensive inference methods, such as MCMC, it is often worthwhile
  ## as in speeds up sampling much more than it slows down setup 

  if (diagonal.penalty && length(sml[[1]]$S)==1) { 
    ## recall that sml is a list that may contain several 'cloned' smooths 
    ## if there was a factor by variable. They have the same penalty matrices
    ## but different model matrices. So cheapest re-para is to use a version
    ## that does not depend on the model matrix (e.g. type=2)
    S11 <- sml[[1]]$S[[1]][1,1];rank <- sml[[1]]$rank;
    p <- ncol(sml[[1]]$X)
    if (is.null(rank) || max(abs(sml[[1]]$S[[1]] - diag(c(rep(S11,rank),rep(0,p-rank))))) > 
        abs(S11)*.Machine$double.eps^.8 ) {
      np <- nat.param(sml[[1]]$X,sml[[1]]$S[[1]],rank=sml[[1]]$rank,type=2,unit.fnorm=FALSE) 
      sml[[1]]$X <- np$X;sml[[1]]$S[[1]] <- diag(p)
      diag(sml[[1]]$S[[1]]) <- c(np$D,rep(0,p-np$rank))
      sml[[1]]$diagRP <- np$P
      if (length(sml)>1) for (i in 2:length(sml)) {
        sml[[i]]$X <- sml[[i]]$X%*%np$P ## reparameterized model matrix
        sml[[i]]$S <- sml[[1]]$S ## diagonalized penalty (unpenalized last)
        sml[[i]]$diagRP <- np$P  ## re-parameterization matrix for use in PredictMat
      }
    } ## end of if, otherwise was already diagonal, and there is nothing to do
  }

  ## The idea here is that term selection can be accomplished as part of fitting 
  ## by applying penalties to the null space of the penalty... 

  if (null.space.penalty) { ## then an extra penalty on the un-penalized space should be added 
    ## first establish if there is a quick method for doing this
    nsm <- length(sml[[1]]$S)
    if (nsm==1) { ## only have quick method for single penalty
      S11 <- sml[[1]]$S[[1]][1,1]
      rank <- sml[[1]]$rank;
      p <- ncol(sml[[1]]$X)
      if (is.null(rank) || max(abs(sml[[1]]$S[[1]] - diag(c(rep(S11,rank),rep(0,p-rank))))) > 
        abs(S11)*.Machine$double.eps^.8 ) need.full <- TRUE else {
        need.full <- FALSE ## matrix is already a suitable diagonal
        if (p>rank) for (i in 1:length(sml)) {
          sml[[i]]$S[[2]] <- diag(c(rep(0,rank),rep(1,p-rank)))
          sml[[i]]$rank[2] <- p-rank
          sml[[i]]$S.scale[2] <- 1 
          sml[[i]]$null.space.dim <- 0
        }
      }
    } else need.full <- if (nsm > 0) TRUE else FALSE

    if (need.full) {
      St <- sml[[1]]$S[[1]]
      if (length(sml[[1]]$S)>1) for (i in 1:length(sml[[1]]$S)) St <- St + sml[[1]]$S[[i]]
      es <- eigen(St,symmetric=TRUE)
      ind <- es$values<max(es$values)*.Machine$double.eps^.66
      if (sum(ind)) { ## then there is an unpenalized space remaining
        U <- es$vectors[,ind,drop=FALSE]
        Sf <- U%*%t(U) ## penalty for the unpenalized components
        M <- length(sm$S)
        for (i in 1:length(sml)) {
          sml[[i]]$S[[M+1]] <- Sf
          sml[[i]]$rank[M+1] <- sum(ind)
          sml[[i]]$S.scale[M+1] <- 1
          sml[[i]]$null.space.dim <- 0
        }
      }
    } ## if (need.full)
  } ## if (null.space.penalty)
  
  if (!apply.by) for (i in 1:length(sml)) {
    by.name <- sml[[i]]$by 
    if (by.name!="NA") {
      sml[[i]]$by <- "NA"
      ## get version of X without by applied...
      sml[[i]]$X0 <- PredictMat(sml[[i]],data)
      sml[[i]]$by <- by.name
    }
  }
  sml
} ## end of smoothCon

# 使用 proto 创建一个新环境，并替换 smoothCon 函数
smoothCon <- with(proto(parent = environment(mgcv::smoothCon), smoothCon = my.smoothCon), smoothCon)

d_s <- read_parquet(args$ipt[1])
k_s <- min(16, floor(args$b/2))
m_s <- bam(nPE ~ te(x, y, k=k_s, bs="bs"), offset=log(nEV), gamma=1.5, nthreads=2, discrete=TRUE, family=poisson(link=log), data=d_s)
print(m_s)

d_t <- read_parquet(args$ipt[2])
k_t <- c(min(16, floor(args$b/2)), min(16, floor(args$b/2)), min(30, floor(args$t/2)))
m_t <- bam(nPE ~ te(x, y, t, k=k_t, bs="bs"), offset=log(nEV)+log(t_binwidth), gamma=3, nthreads=2, discrete=TRUE, family=poisson(link=log), data=d_t)
print(m_t)

models <- list(model_s = m_s, model_t = m_t)
saveRDS(models, args$opt)
