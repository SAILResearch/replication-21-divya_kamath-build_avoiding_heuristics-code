library(dplyr)
library(effsize)
library(FSA)
library(PMCMRplus)
library(DescTools)



build_data <- read.csv('delays.csv')

methods <- list('baseline_SBS_delay', 'baseline_B4_delay', 'baseline_BB_delay', 'baseline_BS4_delay' )
eval_data <- build_data %>% filter(group %in% methods)
kruskal.test(delays ~ group, data = eval_data)

N <- length(eval_data$group)

res <- pairwise.wilcox.test(eval_data$delays, eval_data$group,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra
