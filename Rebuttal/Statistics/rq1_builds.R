library(dplyr)
library(effsize)
library(FSA)
library(PMCMRplus)
library(DescTools)


build_data <- read.csv('builds.csv')

methods <- list('baseline_SBS_builds', 'baseline_B4_builds', 'baseline_BB_builds', 'baseline_BS4_builds' )
eval_data <- build_data %>% filter(group %in% methods)
kruskal.test(builds ~ group, data = eval_data)

N <- length(eval_data$group)

res <- pairwise.wilcox.test(eval_data$builds, eval_data$group,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra
