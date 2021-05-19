library(tidyverse)
library(interactions)
df <- read.csv("parsed_df_210514r.csv") %>% 
    mutate(csse = ifelse(acc==1, sse, NA))



# Main linear regression routine
my_lm_acc <- function(sel_epoch, data, subset_to_strain=F) {
    
    sel_df = filter(data, epoch == sel_epoch)

    if (subset_to_strain) {
        sel_df = filter(sel_df, is_strain_word==1)
    }
    
    m = lm(acc ~ scale(wlen) + scale(log_wf_wsj) * scale(op), data=sel_df)
            
    # Plot interaction for easier interpretation
    print(interact_plot(m, log_wf_wsj, op))
    
    # Print summary
    print(summary(m))
    
    # Return estimates and R2
    estimates <- coef(m)
    
    return(list(
        intercept=estimates['(Intercept)'],
        wlen=estimates['scale(wlen)'],
        wf=estimates['scale(log_wf_wsj)'],
        op=estimates['scale(op)'],
        wfop = estimates['scale(log_wf_wsj):scale(op)'],
        rsq=summary(m)$r.squared))
}


my_lm_csse <- function(sel_epoch, data, subset_to_strain = F) {
    sel_df = filter(data, epoch == sel_epoch) %>% na.omit()
    
    if (subset_to_strain) {
        sel_df = filter(sel_df, is_strain_word == 1)
    }
    
    if (nrow(sel_df) > 100) {
        m = lm(csse ~ scale(wlen) + scale(log_wf_wsj) * scale(op), data = sel_df)
    } else {
        return(list(
            intercept = NA,
            wlen = NA,
            wf = NA,
            op = NA,
            wfop = NA,
            rsq = NA
        ))
    }
    
    # Plot interaction for easier interpretation
    print(interact_plot(m, log_wf_wsj, op))
    
    # Print summary
    print(summary(m))
    
    # Return estimates and R2
    estimates <- coef(m)
    
    return(
        list(
            intercept = estimates['(Intercept)'],
            wlen = estimates['scale(wlen)'],
            wf = estimates['scale(log_wf_wsj)'],
            op = estimates['scale(op)'],
            wfop = estimates['scale(log_wf_wsj):scale(op)'],
            rsq = summary(m)$r.squared
        )
    )
}




epochs <- df$epoch %>% unique() %>% sort()

#### Full training set analysis ####

# ACC
results_train_acc <- map_dfr(epochs, my_lm_acc, data=df) %>% 
    mutate(epoch = epochs) %>% 
    pivot_longer(cols=intercept:rsq, 
                 names_to="parameter", 
                 values_to="std_coef")

# CSSE
results_train_csse <- map_dfr(epochs, my_lm_csse, data=df) %>% 
    mutate(epoch = epochs) %>% 
    pivot_longer(cols=intercept:rsq, 
                 names_to="parameter", 
                 values_to="std_coef")

# Visualize parameters over epoch
my_plot <- function(plot_df, plot_vars){
    
    plot_df_sel <- plot_df %>% filter(parameter %in% plot_vars)
    qplot(x=epoch, y=std_coef, color=parameter, data=plot_df_sel) + 
        facet_grid(.~parameter) +
        theme_minimal()
}



my_plot(results_train_acc, c("intercept", "rsq"))
my_plot(results_train_acc, c("wlen", "op", "wf", "wfop"))

my_plot(results_train_csse, "intercept")
my_plot(results_train_csse, "rsq")
my_plot(results_train_csse, c("wlen", "op", "wf", "wfop"))


#### Strain set analysis ####
results_strain_acc <- map_dfr(epochs, my_lm_acc, df, T) %>% 
    mutate(epoch = epochs) %>% 
    pivot_longer(cols=intercept:rsq, 
                 names_to="parameter", 
                 values_to="std_coef")

my_plot(results_strain_acc, c("intercept", "rsq"))
my_plot(results_strain_acc, c("wlen", "op", "wf", "wfop"))


results_strain_csse <- map_dfr(epochs, my_lm_csse, df, T) %>% 
    mutate(epoch = epochs) %>% 
    pivot_longer(cols=intercept:rsq, 
                 names_to="parameter", 
                 values_to="std_coef")

my_plot(results_strain_csse, "intercept")
my_plot(results_strain_csse, "rsq")
my_plot(results_strain_csse, c("wlen", "op", "wf", "wfop"))

# WF x ACC


tmp <- df %>% 
    filter(epoch == 100) %>% 
    na.omit()
m <- lm(acc ~ scale(wlen) + scale(op), data=tmp)
tmp$rstandard <- rstandard(m)

tmp <- tmp %>% filter(rstandard > -5)

qplot(x=log_wf_wsj, y=rstandard, data=tmp) + 
    geom_smooth() +
    theme_minimal()



## Just correlation over epoch

get_cor <- function(sel_epoch, use_df){
    tmp <- use_df %>% 
        filter(epoch==sel_epoch) %>% 
        select(log_wf_wsj, sse) %>% 
        na.omit() %>% 
        cor()
    
    return(list(r=tmp[2]))
}


rs <- map_dfr(epochs, get_cor, use_df=df) %>% 
    mutate(epoch = epochs) 

qplot(x=epoch, y=r, data=rs) + geom_line()
