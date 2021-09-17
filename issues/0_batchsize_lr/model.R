library(tidyverse)
library(effectsize)


# Theoretical fluctuation or effective change in the gradient
# Somewhat similar to epsilon / batch_size if batch size << trainset_size

get_f <- function(epsilon, trainset_size, batch_size){
    return (((trainset_size/batch_size) - 1)*epsilon)
}

df <- read.csv("taraban80.csv") %>% 
    mutate(f = get_f(learning_rate, 5861, batch_size))

mdf <- df %>% 
    group_by(code_name, batch_size, learning_rate, freq, reg) %>% 
    summarise(macc = mean(acc), msse=mean(sse))

glm(macc~learning_rate + batch_size + freq + reg, data=mdf) %>% 
    standardize_parameters()









# Check is f able to explain everything
glm(acc ~ f + batch_size * learning_rate, family="binomial", data=df) %>% 
    standardize_parameters()

# Maybe cannot reduce to a single dimension

m2 <- glm(acc~ batch_size * freq * reg, family="binomial", data=df)
standardize_parameters(m2)





ggplot()

my_glm <- function(f, data){
    
    m <- glm(as.formula(f), family = "binomial", data=data)
    s <- standardize_parameters(m, method = "basic")
    print(s)
    return(list(s$Parameter, s$Std_Coefficient))
    
}

glm_tarban_firstlv <- function(data){
    m <- glm(acc ~ freq * reg, family = "binomial", data)
    return(standardize_parameters(m))
}


glm_tarban_firstlv(df %>% filter(code_name=='task_effect_r0017'))
