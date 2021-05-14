library(tidyverse)
library(interactions)
df = read.csv("parsed_df_210514r.csv")

# Main linear regression routine
my_lm <- function(sel_epoch, data, subset_to_strain=F) {
    
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


epochs <- df$epoch %>% unique() %>% sort()

#### Full training set analysis ####
results_train <- map_dfr(epochs, my_lm, data=df) %>% 
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

var_set1 <- c("intercept", "rsq")
var_set2 <- c("wlen", "op", "wf", "wfop")

my_plot(results_train, var_set1)
my_plot(results_train, var_set2)

#### Strain set analysis ####
results_strain <- map_dfr(epochs, my_lm, df, T) %>% 
    mutate(epoch = epochs) %>% 
    pivot_longer(cols=intercept:rsq, 
                 names_to="parameter", 
                 values_to="std_coef")

my_plot(results_strain, var_set1)
my_plot(results_strain, var_set2)

