library(dplyr)
library(effsize)
library(FSA)
library(PMCMRplus)
library(DescTools)


delay_data <- read.csv('delays.csv')
tr_delays <- list('tr_B4_delay', 'tr_BB_delay', 'tr_BS4_delay')
b4_groups <- list('tr_B4_delay', 'baseline_B4_delay')
bb_groups <- list('tr_BB_delay', 'baseline_BB_delay')
bs4_groups <- list('tr_BS4_delay', 'baseline_BS4_delay')

print('Amongst Timeout Rule')
tr_data <- delay_data %>% filter(group %in% tr_delays)
kruskal.test(delays ~ group, data = tr_data)

N <- length(tr_data$group)
res <- pairwise.wilcox.test(tr_data$delays, tr_data$group,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra

sbs_delays <- delay_data %>% filter((group == 'baseline_SBS_delay') | (group %in% tr_delays))

print('Smartdelayskip v/s tr')
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
  
  cat('Evaluating ', tr_delays[[i]], group_list[[2]])
  
  baseline <- group_data %>% filter(group == group_list[2])
  hybrid <- group_data %>% filter(group == group_list[1])
  
  res <- wilcox.test(baseline$delays, hybrid$delays,p.adjust.method = "BH")
  print(res)
  Za = qnorm(res$p.value/2)
  ra = abs(Za)/sqrt(N)
  print(ra)
  
}