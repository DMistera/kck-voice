def fund (
                  wave,
                  f,
                  wl = 512,
                  ovlp = 0,
                  fmax = f/2,
                  threshold = NULL, 
                  at = NULL,
                  from = NULL,
                  to = NULL,
                  plot = TRUE,
                  xlab = "Time (s)",
                  ylab = "Frequency (kHz)", 
                  ylim = c(0, f/2000),
                  pb = FALSE
                  ):
        ## ERROR MESSAGES
        if(!is.null(at)){
        if(!is.null(threshold)) stop("The 'threshold' argument cannot be used with the argument 'at'.")
        if(ovlp!=0) stop("The 'overlap' argument should equal to 0 when using the argument 'at'.")
        if(!is.null(from) | !is.null(to)) stop("The 'from' and/or 'to' arguments cannot be used when using the argument 'at'.")
        if(pb) stop("No progress bar can be displayed when using the argument 'at'.")
        if(plot) {
            plot <- FALSE
            warning("When the argument 'at' is used, the argument 'plot' is automatically turned to 'FALSE'.")}
        }
        
        ## INPUT
        input <- inputw(wave = wave, f = f)
        wave <- input$w
        f <- input$f
        rm(input)
        WL <- wl%/%2

        ## FROM-TO SELECTION
        if (!is.null(from) | !is.null(to)) {
            if (is.null(from) && !is.null(to)) {
                a <- 1
                b <- round(to * f)
            }
            if (!is.null(from) && is.null(to)) {
                a <- round(from * f)
                b <- length(wave)
            }
            if (!is.null(from) && !is.null(to)) {
                if (from > to) 
                    stop("'from' cannot be superior to 'to'")
                if (from == 0) {
                    a <- 1
                }
                else {
                    a <- round(from * f)
                }
                b <- round(to * f)
            }
            wave <- as.matrix(wave[a:b, ])
        }

        ## AT SELECTION
        if (!is.null(at)) {
            c <- round(at * f)
            wave <- as.matrix(wave[(c - WL):(c + WL), ]) 
        }
        
        ## THRESHOLD
        if (!is.null(threshold)) {
            wave <- afilter(wave = wave, f = f, threshold = threshold, 
                            plot = FALSE)
        }

        ## SLIDING WINDOW
        wave <- ifelse(wave == 0, yes = 1e-06, no = wave)
        n <- nrow(wave)
        step <- seq(1, n+1-wl, wl-(ovlp * wl/100)) # +1 added @ 2017-04-20
        N <- length(step)
        z1 <- matrix(data = numeric(wl * N), wl, N)

        ## CEPSTRUM AND OPTIONAL PROGRESS BAR
        if (pb) {
            pbar <- txtProgressBar(min = 0, max = n, style = 3)
        }
        for (i in step) {
            z1[, which(step == i)] <- Re(fft(log(abs(fft(wave[i:(wl + 
                           i - 1), ]))), inverse = TRUE))
            if (pb) {
                setTxtProgressBar(pbar, i)
            }
        }

        ## FUNDAMENTAL FREQUENCY TRACKING
        z2 <- z1[1:WL, ]
        z <- ifelse(z2 == "NaN" | z2 == "-Inf", yes = 0, no = z2)
        z <- as.matrix(z)
        fmaxi <- f%/%fmax
        tfund <- numeric(N)
        for (k in 1:N) {
            tfund[k] <- which.max(z[-c(1:fmaxi), k])
        }
        tfund <- as.numeric(ifelse(tfund == 1, yes = NA, no = tfund))  ## "NA" passé en NA
        ffund <- f/(tfund + fmaxi - 1)
        if(!is.null(at)) {x <- at} else {x <- seq(0, n/f, length.out = N)}
        y <- ffund/1000
        res <- cbind(x, y)

        ## PLOT AND RETURN
        if (plot) {
            plot(x = x, y = y,
                 xaxs = "i", xlab = xlab,
                 yaxs = "i", ylab = ylab, ylim = ylim,
                 las = 1, ...)
            invisible(return(res))    # necessary to include return() otherwise nothing is returned
        }       
        else {return(res)}
        if (pb) close(pbar)
    }
