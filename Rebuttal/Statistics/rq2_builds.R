library(dplyr)
library(effsize)
library(FSA)
library(PMCMRplus)
library(DescTools)


build_data <- read.csv('builds.csv')
mlci_builds <- list('mlci_B4_builds', 'mlci_BB_builds', 'mlci_BS4_builds')
b4_groups <- list('mlci_B4_builds', 'baseline_B4_builds')
bb_groups <- list('mlci_BB_builds', 'baseline_BB_builds')
bs4_groups <- list('mlci_BS4_builds', 'baseline_BS4_builds')

print('Amongst MLCI')
mlci_data <- build_data %>% filter(group %in% mlci_builds)
kruskal.test(builds ~ group, data = mlci_data)

N <- length(mlci_data$group)
res <- pairwise.wilcox.test(mlci_data$builds, mlci_data$group,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra

sbs_builds <- build_data %>% filter((group == 'baseline_SBS_builds') | (group %in% mlci_builds))

print('SmartBuildSkip v/s MLCI')
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
  
  cat('Evaluating ', mlci_builds[[i]], group_list[[2]])
  
  baseline <- group_data %>% filter(group == group_list[2])
  hybrid <- group_data %>% filter(group == group_list[1])
  
  res <- wilcox.test(baseline$builds, hybrid$builds,p.adjust.method = "BH")
  print(res)
  Za = qnorm(res$p.value/2)
  ra = abs(Za)/sqrt(N)
  print(ra)
  
}


