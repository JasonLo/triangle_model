library(tidyverse)
library(gridExtra)

input_s_v1 <- function(e, t, f, g, k, tmax){
  numer_f <- g * e * log(f+2)
  denom_f <- e * log(f+2) + k 
  input_s_plaut <- (t/tmax)*(numer_f/denom_f)
}

sigmoid <- function(x){
  return(1/(1+exp(-(x))))
}


# Plotting Wrapper
plot_s <- function(g, k){

  strain_high_frequency = 7700
  strain_low_frequency = 402
  
  e = 1:1000
  
  # At last time step t=3.8 (zero indexing in python...)
  s_lowf <- input_s_v1(e, t=3.8, f=strain_low_frequency, g=g, k=k, tmax=3.8)
  s_highf <- input_s_v1(e, t=3.8, f=strain_high_frequency, g=g, k=k, tmax=3.8)
  df <- data.frame(e, s_lowf, s_highf)
  df <- df %>% 
    mutate(a_lowf = sigmoid(s_lowf)) %>%
    mutate(a_highf = sigmoid(s_highf))
  
  input <- ggplot(df) + 
    geom_line(aes(e, s_lowf), color = 'red') +
    geom_line(aes(e, s_highf), color = 'blue') +
    ylab('Semantic input to P') +
    xlab('Epoch')
  
  act <- ggplot(df)  + 
    geom_line(aes(e, a_lowf), color = 'red') +
    geom_line(aes(e, a_highf), color = 'blue') +
    ylab('Semantic activation to P') +
    xlab('Epoch')
  
  grid.arrange(input, act, ncol=2)
  
}

# Play with g and k...
plot_s(g=20, k=5000)


