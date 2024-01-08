library(dplyr)
library(effsize)
library(FSA)
library(PMCMRplus)
library(DescTools)


build_data <- read.csv('builds.csv')
tr_builds <- list('tr_B4_builds', 'tr_BB_builds', 'tr_BS4_builds')
b4_groups <- list('tr_B4_builds', 'baseline_B4_builds')
bb_groups <- list('tr_BB_builds', 'baseline_BB_builds')
bs4_groups <- list('tr_BS4_builds', 'baseline_BS4_builds')

print('Amongst Timeout Rule')
tr_data <- build_data %>% filter(group %in% tr_builds)
kruskal.test(builds ~ group, data = tr_data)

N <- length(tr_data$group)
res <- pairwise.wilcox.test(tr_data$builds, tr_data$group,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra


sbs_builds <- build_data %>% filter((group == 'baseline_SBS_builds') | (group %in% tr_builds))

print('SmartBuildSkip v/s tr')
kruskal.test(builds ~ group, data = sbs_builds)

N <- length(sbs_builds$group)
res <- pairwise.wilcox.test(sbs_builds$builds, sbs_builds$group,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra

batching_lists <- list(b4_groups, bb_groups, bs4_groups)
list_range <- 1:3

for (i in list_range) {
  
  group_list <- batching_lists[[i]]
  group_data <- build_data %>% filter(group %in% group_list)
  
  cat('Evaluating ', tr_builds[[i]], group_list[[2]])
  
  baseline <- group_data %>% filter(group == group_list[2])
  hybrid <- group_data %>% filter(group == group_list[1])
  
  res <- wilcox.test(baseline$builds, hybrid$builds,p.adjust.method = "BH")
  print(res)
  Za = qnorm(res$p.value/2)
  ra = abs(Za)/sqrt(N)
  print(ra)
  
}


