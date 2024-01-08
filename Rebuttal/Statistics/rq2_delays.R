library(dplyr)
library(effsize)
library(FSA)
library(PMCMRplus)
library(DescTools)


delay_data <- read.csv('delays.csv')
mlci_delays <- list('mlci_B4_delay', 'mlci_BB_delay', 'mlci_BS4_delay')
b4_groups <- list('mlci_B4_delay', 'baseline_B4_delay')
bb_groups <- list('mlci_BB_delay', 'baseline_BB_delay')
bs4_groups <- list('mlci_BS4_delay', 'baseline_BS4_delay')

print('Amongst MLCI')
mlci_data <- delay_data %>% filter(group %in% mlci_delays)
kruskal.test(delays ~ group, data = mlci_data)

N <- length(mlci_data$group)
res <- pairwise.wilcox.test(mlci_data$delays, mlci_data$group,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra

sbs_delays <- delay_data %>% filter((group == 'baseline_SBS_delay') | (group %in% mlci_delays))

print('Smartdelayskip v/s MLCI')
kruskal.test(delays ~ group, data = sbs_delays)

N <- length(sbs_delays$group)
res <- pairwise.wilcox.test(sbs_delays$delays, sbs_delays$group,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra

batching_lists <- list(b4_groups, bb_groups, bs4_groups)
list_range <- 1:3

for (i in list_range) {
  
  group_list <- batching_lists[[i]]
  group_data <- delay_data %>% filter(group %in% group_list)
  
  cat('Evaluating ', mlci_delays[[i]], group_list[[2]])
  
  baseline <- group_data %>% filter(group == group_list[2])
  hybrid <- group_data %>% filter(group == group_list[1])
  
  res <- wilcox.test(baseline$delays, hybrid$delays,p.adjust.method = "BH")
  print(res)
  Za = qnorm(res$p.value/2)
  ra = abs(Za)/sqrt(N)
  print(ra)
  
}