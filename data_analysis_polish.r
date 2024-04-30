library(dplyr)
library(ggplot2)
library(ggpubr)
library(RColorBrewer)
library(broom)
library(ggrepel)
library(scales)
# library(Hmisc)
library(tidyr)
library(stringr)
library(psych)
library(tidyverse)
library(stats)

sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}

logit <- function(p){
  epsilon = 1e-15
  return(log((p + epsilon) / ((1 - p) + epsilon)))
}

tanh <- function(x){
  return(tanh(x))
}

eharmony_function <- function(x){
  return(exp(-x))
}

laplace_correction <- function(counter, k=1){
  return(counter + k)
}

log_corrected_counts <- function(counter){
  return(logit(counter))
}

acceptability <- function(score){
  corrected_counts = laplace_correction(score)
  log_counts = log_corrected_counts(corrected_counts)
  return(log_counts)
}


get_attestedness <- function(typc){
  if (typc == "na"){
    return('unattested')
  } else if (as.numeric(typc) < 10){
    return('0 < type < 10')
  } else {
    return('type > 10')
  }
}

goodman_kruskals_gamma <- function(x, y) {
  concordant <- 0
  discordant <- 0
  ties <- 0
  
  for (i in 1:(length(x) - 1)) {
    for (j in (i+1):length(x)) {
      if ((x[i] > x[j] && y[i] > y[j]) || (x[i] < x[j] && y[i] < y[j])) {
        concordant <- concordant + 1
      } else if ((x[i] > x[j] && y[i] < y[j]) || (x[i] < x[j] && y[i] > y[j])) {
        discordant <- discordant + 1
      } else {
        ties <- ties + 1
      }
    }
  }
  
  cat("Number of concordant:", concordant, "\n")
  cat("Number of discordant:", discordant, "\n")
  cat("Number of ties:", ties, "\n")
  
  gamma <- (concordant - discordant) / (concordant + discordant)
  # tau_a <- (concordant - discordant) / (concordant + discordant + ties)
  
  return(gamma)
}

setwd("~/Documents/projects/exception-filtering-learner")

# polish

input_path <- "result/polish/judgment_model-filtering_struc-local_flt-T_pad-F_conf-0.975_pen-10_thr-0.1.txt" #polishJudgement-exception-filtering-final
input_path <- "result/polish/judgment_model-gross_struc-local_flt-T_pad-F_conf-0.975_pen-3_thr-0.2.txt" # gross
output_path <- "result/polish/correlation_plot.png"
output_path_ucla <- "result/polish/correlation_plot_ucla.png"

data <- read.csv(input_path, sep='\t', header = FALSE, na.strings = c("NA", "na", "#N/A"))

colnames(data) <- c("form","tail", "ucla", "subj", "likert_rating", "tokcount", "typc", "score")

# merge to the original testing file for more random effects
df <- read.csv("data/polish/all_legit_data.csv", stringsAsFactors = FALSE)
# Extract the "ssp" column to a new dataframe
data$ssp <- as.numeric(df$ssp)
# data$head <- as.character(df$head)
# data$head <- as.character(df$head)
# 
data$rt <- as.numeric(df$rt)
data$novel <- as.numeric(df$novel)

replace_dict <- c('rz' = 'ʐ', 'sz' = 'ʂ', 'cz' = 'tʂ', 'dzi' = 'dʑ', 'si' = 'ɕ', 'zi' = 'ʑ', 'ni' = 'ɲ', 'dz' = 'dz',  'w' = 'v', 'ł' = 'w', 'ch' = 'x')
# cz -> tʂ -> ʧ
# drz -> d͡ʐ -> ʤ
# sz -> ʂ
# ż -> ʐ
data <- data %>%
  mutate(form = str_replace_all(form, replace_dict))

data$likert_rating <- as.factor(data$likert_rating)
data$score <- as.numeric(as.character(data$score))
data$typc <- as.numeric(data$typc)
table(is.na(as.numeric(as.character(data$typc))))
data$typc[is.na(data$typc)] <- 0

data$attestedness <- ifelse(as.numeric(data$typc) == 0, 0, 1)

# Replicate Jarosz & Rysling (2017)
# Load necessary packages
library(ordinal)

# Assuming your data is in a dataframe named df
# and the columns are named rating, SSP, attestedness, participant, and tail

# Fit the model
# model <- clmm(likert_rating ~ ssp * score + (1 + ssp | subj) + (1 | tail), data = data)
# model <- clmm(likert_rating ~ score + (1|subj), data = data)
# model <- clmm(likert_rating ~ score + (1|subj) + (1 | tail) + (1|typc) + (1|form) + (1|novel) +(1|rt) , data = data)
# # model <- clmm(likert_rating ~ score + ssp * attestedness + typc + (1 + ssp + attestedness | subj) + (1 + ssp + attestedness | tail), data = data)
# model <- clmm(likert_rating ~ score + typc + (1|subj) + (1|tail), data = data)



# Standardize continuous predictors if needed
data$score <- scale(data$score)
data$ssp <- scale(data$ssp)
data$typc <-scale(data$typc)


# model <- clmm(likert_rating ~ score + typc + ssp * attestedness + (1 + ssp + attestedness|subj) + (1 + ssp + attestedness|tail), data = data)

# model <- clmm(likert_rating ~ score + typc + ssp * attestedness + 
#                 (1|subj) + (1|tail), data = data)
# 
# # Print the summary
# summary(model)


# 1. find underlying 0 and 1 categorical grammaticality


# HW MaxEnt learner
maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/Polish/n=2/07maxoe-basic/blickTestResults.txt"
maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/Polish/n=2/07maxoe-basic-sigma1/blickTestResults.txt"
# maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/Polish/n=2/07maxoe-basic-sigma01/blickTestResults.txt"

# maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/Polish/n=2/07maxoe-basic-complementation/blickTestResults.txt"
maxent_data <- read_delim(maxent_path, delim = '\t', skip = 2)

maxent_data <- maxent_data[, 1:2]  # Keep only the first two columns

# Rename the "score" column to "harmony"
names(maxent_data)[names(maxent_data) == "score"] <- "harmony"

data <- cbind(data, maxent_data)
data$ucla <- data$harmony
data$eharmony <- sapply(data$ucla, eharmony_function)
# data$attestedness <- sapply(data$typc, get_attestedness)

# Create a new column 'form_tail' that combines 'form' and 'tail'
# data$form_tail <- paste(data$form, data$tail, sep = "_")

# Calculate standard deviation grouped by 'form_tail'
# likert_sd <- aggregate(likert_rating ~ form_tail, data, sd)

# likert_sd <- aggregate(likert_rating ~ form, data, sd)

# par(mar = c(2, 4, 2, 2))
# set.seed(123)  # for reproducibility
# sample_data <- data[sample(1:nrow(data), 5000), ]
# shapiro.test(sample_data$likert_rating)  # for Shapiro-Wilk test

# hist(data$likert_rating)  # for histogram
# qqnorm(data$likert_rating)  # for Q-Q plot
# qqline(data$likert_rating)  # adds a 45-degree reference line to the Q-Q plot

# library(lme4)

# Assuming your data is in a dataframe called 'data' and the columns are named 'rating', 'score', 'participant', 'onset':

# test$likert_rating <- ordered(test$likert_rating, levels = c("1", "2", "3", "4", "5", "6"))  # adjust the levels according to your Likert scale
# mod <- polr(likert_rating ~ eharmony, data = test, Hess = TRUE)

# mod <- polr(as.ordered(likert_rating) ~ score, data = data, Hess=TRUE)

# library(MASS)
# # Fit the ordinal logistic regression model
# mod1 <- polr(as.ordered(likert_rating) ~ eharmony, data = data, Hess=TRUE)
# mod1 <- polr(as.ordered(likert_rating) ~ score, data = data, Hess=TRUE)
# library("brant")
# 
# brant(mod1)


# model <- clmm(likert_rating ~ eharmony + typc + ssp * attestedness + 
#                 (1|subj) + (1|tail), data = data)
# 
# model <- clmm(likert_rating ~ eharmony  + ssp * attestedness + 
#                 (1|subj) + (1|tail), data = data)
# model <- clmm(likert_rating ~ score  + ssp * attestedness + 
#                 (1|subj) + (1|tail), data = data)
# model <- clmm(likert_rating ~ eharmony+ (1|subj) + (1|tail), data = data)
# 
# model <- clmm(likert_rating ~ eharmony + typc +  (1|subj) + (1|tail), data = data)


model <- clmm(likert_rating ~ score + typc +  (1|subj) + (1|tail), data = data)
# Print the summary
summary(model)

# Show summary of the model
# summary(mod1)

# Then you can compute AIC and BIC as usual
# aic <- AIC(mod1)
# bic <- BIC(mod1)
# 
# 
# print(paste("AIC: ", aic))
# print(paste("BIC: ", bic))

# data$likert_rating_zscore <- scale(data$likert_rating)
# standard deviation per form over all subjects
data$likert_rating <- as.numeric(as.character(data$likert_rating))

data <- data %>%
  group_by(form) %>%
  summarise(score = mean(score),
            ucla = mean(ucla),
            ssp = mean(ssp),
            likert_rating = mean(likert_rating), 
            # likert_rating_zscore = mean(likert_rating_zscore),
            eharmony = mean(eharmony),
            attestedness = mean(attestedness)
            ) %>%
  ungroup()

# model <- lm(likert_rating ~ score, data = data)
# model <- lm(likert_rating ~ eharmony, data = data)

# print the summary
# summary(model)

#AIC BIC
# Fit a simple linear model
# model <- lm(likert_rating ~ score, data = data)

# Now, you can compute AIC and BIC
# aic <- AIC(model)
# bic <- BIC(model)
# 
# # Print them out
# print(paste("AIC: ", aic))
# print(paste("BIC: ", bic))


correlation_methods <- c('pearson', 'spearman', 'kendall')
for (method in correlation_methods){
  s = cor(data$score, data$likert_rating, method=method)
  print(paste("NT: ", method, ": ", round(s, 3)))
  s = cor(data$score, data$ssp, method=method)
  print(paste("NT-SSP: ", method, ": ", round(s, 3)))
  s = cor(data$ssp, data$likert_rating, method=method)
  print(paste("SSP-likert ", method, ": ", round(s, 3)))
  s = cor(data$eharmony, data$likert_rating, method=method)
  print(paste("UCLA: ", method, ": ", round(s, 3)))
  s = cor(data$eharmony, data$ssp, method=method)
  print(paste("UCLA-SSP: ", method, ": ", round(s, 3)))
  }

# Separate the data into two groups based on attestedness
data_attested <- data[data$attestedness == 1, ]
data_unattested <- data[data$attestedness == 0, ]

# Compute correlations for each group
for (method in correlation_methods) {
  # For attested data
  print(paste("Attended data correlations with method:", method))
  s = cor(data_attested$score, data_attested$likert_rating, method=method)
  print(paste("NT: ", round(s, 3)))
  s = cor(data_attested$eharmony, data_attested$likert_rating, method=method)
  print(paste("UCLA: ", round(s, 3)))
#   
  # For unattested data
  print(paste("Unattested data correlations with method:", method))
  s = cor(data_unattested$score, data_unattested$likert_rating, method=method)
  print(paste("NT: ", round(s, 3)))
  s = cor(data_unattested$eharmony, data_unattested$likert_rating, method=method)
  print(paste("UCLA: ", round(s, 3)))
}

gamma_result <- goodman_kruskals_gamma(data$likert_rating, data$score)
print(paste("NT Goodman Kruskal's Gamma: ", gamma_result))
gamma_result <- goodman_kruskals_gamma(data$likert_rating, data$eharmony)
print(paste("UCLA Goodman Kruskal's Gamma: ", gamma_result))



gamma_result <- goodman_kruskals_gamma(data_attested$likert_rating, data_attested$score)
print(paste("NT Goodman Kruskal's Gamma: ", gamma_result))
gamma_result <- goodman_kruskals_gamma(data_unattested$likert_rating, data_unattested$eharmony)
print(paste("UCLA Goodman Kruskal's Gamma: ", gamma_result))


# Categorical accuracy
# Test data and predictions

data$likert_rating_binary <- ifelse(data$likert_rating >= 0, 1, 0)


# Categorical
# Binary accuracy  
data %>%
  mutate(correct_prediction = ifelse(likert_rating_binary == score, 1, 0)) %>%
  summarise(accuracy = mean(correct_prediction))

data %>%
  mutate(correct_prediction = ifelse(likert_rating_binary == score, 1, 0)) %>%
  group_by(likert_rating_binary) %>%
  summarise(accuracy = mean(correct_prediction))

incorrect_predictions <- data %>%
  mutate(correct_prediction = ifelse(likert_rating_binary == score, 1, 0)) %>%
  filter(correct_prediction == 0)

print(incorrect_predictions)

# F-score for Categorical

# Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
TP <- sum(data$score == 1 & data$likert_rating_zscore >= 0)
FP <- sum(data$score == 1 & data$likert_rating_zscore < 0)
FN <- sum(data$score == 0 & data$likert_rating_zscore >= 0)

# Calculate Precision, Recall
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

# Calculate F1 Score
F1 <- 2 * (precision * recall) / (precision + recall)

# Print the F1-score
print(F1)
print(precision)
print(recall)

# HW learner
optimize_theta <- function(data) {
  best_theta <- 0
  best_accuracy <- -Inf
  unique_values <- sort(unique(data$eharmony))
  
  for(theta in unique_values) {
    data %>% mutate(
      accept = (eharmony >= theta),
      acc = ifelse(likert_rating_zscore > 0, accept, 1-accept)) ->
      data
    
    data$likert_rating_binary <- ifelse(data$likert_rating_zscore >= 0, 1, 0)
    
    accuracy <- data %>%
      summarise(acc = mean(acc))
    
    if (accuracy > best_accuracy) {
      best_accuracy <- accuracy
      best_theta <- theta
    }
  }
  print(paste('best accuracy:', best_accuracy))
  print(paste('best theta:', best_theta))
  
  return(best_theta)
}
theta = optimize_theta(data)


# F-score
# theta = with(subset(data, likert_rating_zscore > 0), quantile(eharmony, 0.99, na.rm=TRUE))
# theta = 0.01


# Precision and recall
precision = with(subset(data, eharmony >= theta), mean(likert_rating_zscore > 0))
recall = with(subset(data, likert_rating_zscore > 0), mean(eharmony >= theta))
F = 2 * (precision * recall) / (precision + recall)
F
recall
precision



# Binary accuracy
data %>% mutate(
  accept = (eharmony >= theta),
  acc = ifelse(likert_rating_zscore > 0, accept, 1-accept)) ->
  data

data$likert_rating_binary <- ifelse(data$likert_rating_zscore >= 0, 1, 0)

data %>%
  summarise(acc = mean(acc))

data %>%
  group_by(likert_rating_binary) %>%
  summarise(acc = mean(acc))

data$likert_rating_binary <- ifelse(data$likert_rating_zscore > 0, 1, 0)
incorrect_predictions <- data %>%
  mutate(correct_prediction = ifelse(acc == 1, 1, 0)) %>%
  filter(correct_prediction == 0)

print(incorrect_predictions)

# Plotting

spearman_corr <- cor.test(data$score, data$likert_rating, method="spearman")
# annotation_spearman <- paste("Spearman: ", round(spearman_corr, 3))


# Rename the model names and set factor levels
data_long <- data %>%
  pivot_longer(cols = c(eharmony, score),
               names_to = "model",
               values_to = "value") %>%
  mutate(model = case_when(
    model == "eharmony" ~ "H&W",
    model == "score" ~ "NT",
    TRUE ~ model
  ))



# Calculate Spearman correlation for each model and store in data frame
correlations <- data_long %>%
  group_by(model) %>%
  summarize(correlation = round(cor.test(value, likert_rating, method="spearman")$estimate, 3)) %>%
  mutate(model_correlation = paste0(model, ": ", round(correlation, 3)))

# Join correlations to the original data frame
data_long <- data_long %>%
  left_join(correlations, by = "model")

# Set the factor levels to include the correlation in the facet labels
# Ensure that the "NT" level comes before the "H&W" level
data_long$model_correlation <- factor(data_long$model_correlation, levels = c(paste0("NT: ", round(correlations$correlation[correlations$model == "NT"], 3)), paste0("H&W: ", round(correlations$correlation[correlations$model == "H&W"], 3))))

# Plot with annotation
averaged_plot <- ggplot(data_long, aes(x=value, y=likert_rating)) +
  geom_smooth(method='lm', aes(x=value, y=likert_rating), color = 'gray') +
  geom_point(aes(color=attestedness, shape=attestedness)) + 
  geom_text_repel(aes(label=form, color=attestedness),  show.legend=FALSE, size=4) +
  scale_color_brewer(type="qual", palette="Set1") +
  labs(x='Predicted judgment', y='Likert rating') +
  facet_wrap(~ model_correlation, scales = "free_x", nrow = 2) + # set nrow to 2
  theme_classic() +
  theme(
    legend.position=c(0.87, 0.15),
    legend.direction='vertical',
    legend.title=element_blank(),
    text=element_text(family="Times New Roman", color="black", size=14)
  )

averaged_plot <- ggplot(data_long, aes(x=likert_rating, y=value)) +
  geom_smooth(method='lm', aes(x=likert_rating, y=value), color = 'gray') +
  geom_point(aes(color=attestedness, shape=attestedness)) + 
  geom_text_repel(aes(label=form, color=attestedness),  show.legend=FALSE, size=4) +
  scale_color_brewer(type="qual", palette="Set1") +
  labs(x='Predicted judgment', y='Likert rating') +
  facet_wrap(~ model_correlation, scales = "free_x", nrow = 2) + # set nrow to 2
  theme_classic() +
  theme(
    legend.position=c(0.87, 0.15),
    legend.direction='vertical',
    legend.title=element_blank(),
    text=element_text(family="Times New Roman", color="black", size=14)
  )
# Save the plot
ggsave(filename = "models_plot.png", plot = averaged_plot, dpi = 300)

averaged_plot

# english
input_path <- "result/english/EnglishJudgement_filtering_best.txt"
input_path <- "result/english/EnglishJudgement_gross.txt"

test_path <- "data/english/Daland_etal_2011_AverageScores.csv"
output_path <- "plot/english/correlation_plot.png"

test <- read.csv(test_path, sep=',', header = TRUE)
pred <- read.csv(input_path, sep='\t', header = FALSE)
colnames(pred) <- c("form","attestedness", "grammaticality")


test <- cbind(test, pred)
# Get the names of duplicate columns
duplicate_columns <- names(test)[duplicated(names(test))]
# Remove duplicate columns
test <- test[ , !duplicated(names(test))]

test <- test %>%
  group_by(form) %>%
  summarise(
    grammaticality = mean(grammaticality),
    likert_rating = mean(likert_rating),
  ) %>%
  ungroup()
# Test data and predictions
test$likert_rating_zscore <- scale(test$likert_rating)

test$likert_rating_binary <- ifelse(test$likert_rating_zscore >= 0, 1, 0)

# Dai learner
# Binary accuracy
test %>%
  mutate(correct_prediction = ifelse(likert_rating_binary == grammaticality, 1, 0)) %>%
  summarise(accuracy = mean(correct_prediction))

test %>%
  mutate(correct_prediction = ifelse(likert_rating_binary == grammaticality, 1, 0)) %>%
  group_by(likert_rating_binary) %>%
  summarise(accuracy = mean(correct_prediction))

incorrect_predictions <- test %>%
  mutate(correct_prediction = ifelse(likert_rating_binary == grammaticality, 1, 0)) %>%
  filter(correct_prediction == 0)

print(incorrect_predictions)


# HW MaxEnt learner
# Test data and predictions
maxent_path <- "result/english/EnglishJudgement_MaxEnt_OE03_unlimited.txt"
test <- read.csv(test_path, sep=',', header = TRUE)
pred <- read.csv(maxent_path, sep='\t', header = TRUE)
colnames(pred) <- c('form',	'harmony')

# pred <- read_tsv(maxent_path, col_names = c('form',	'harmony'))

test <- cbind(test, pred)
test <- test %>%
  select(form, harmony, likert_rating)

test <- test %>%
  mutate(eharmony = eharmony_function(harmony)) %>%
  group_by(form) %>%
  summarise(
    harmony = mean(harmony),
    eharmony = mean(eharmony),
    likert_rating = mean(likert_rating)
  ) %>%
  ungroup()

correlation_methods <- c('pearson', 'spearman', 'kendall')
for (method in correlation_methods){
  s = cor(test$eharmony, test$likert_rating, method=method)
  print(paste("UCLA: ", method, ": ", round(s, 3)))
}

goodman_kruskals_gamma <- function(x, y) {
  concordant <- 0
  discordant <- 0
  ties <- 0
  
  for (i in 1:(length(x) - 1)) {
    for (j in (i+1):length(x)) {
      if ((x[i] > x[j] && y[i] > y[j]) || (x[i] < x[j] && y[i] < y[j])) {
        concordant <- concordant + 1
      } else if ((x[i] > x[j] && y[i] < y[j]) || (x[i] < x[j] && y[i] > y[j])) {
        discordant <- discordant + 1
      } else {
        ties <- ties + 1
      }
    }
  }
  
  cat("Number of concordant:", concordant, "\n")
  cat("Number of discordant:", discordant, "\n")
  cat("Number of ties:", ties, "\n")
  
  gamma <- (concordant - discordant) / (concordant + discordant)
  tau_a <- (concordant - discordant) / (concordant + discordant + ties)
  
  return(gamma)
}

gamma_result <- goodman_kruskals_gamma(test$likert_rating, test$eharmony)
print(paste("Goodman Kruskal's Gamma: ", gamma_result))

# # F-score
# theta = with(subset(test, likert_rating > 3.5), quantile(eharmony, 0.99, na.rm=TRUE))
# # how many harmony
# precision = with(subset(test, eharmony >= theta),
#                  mean(likert_rating >  3.5))
# recall = with(subset(test, likert_rating >  3.5),
#               mean(eharmony >= theta))
# F = 2 * (precision * recall) / (precision + recall)
# 
# # Binary accuracy
# test %>% mutate(
#   accept = (eharmony >= theta),
#   acc = ifelse(likert_rating >  3.5, accept, 1-accept)) ->
#   test
# 
# test$likert_rating_binary <- ifelse(test$likert_rating >= 3.5, 1, 0)
# 
# test %>%
#   summarise(acc = mean(acc))
# 
# test %>%
#   group_by(likert_rating_binary) %>%
#   summarise(acc = mean(acc))
# # 
# # test %>%
# #   filter(likert_rating_binary == 0) %>%
# #   arrange(-harmony) %>%
# #   head()
# 
# test$likert_rating_binary <- ifelse(test$likert_rating >  3.5, 1, 0)
# incorrect_predictions <- test %>%
#   mutate(correct_prediction = ifelse(acc == 1, 1, 0)) %>%
#   filter(correct_prediction == 0)
# 
# print(incorrect_predictions)

test$likert_rating_zscore <- scale(test$likert_rating)
# F-score
theta = with(subset(test, likert_rating_zscore >= 0), quantile(eharmony, 0.99, na.rm=TRUE))
theta
# Precision and recall
precision = with(subset(test, eharmony >= theta), mean(likert_rating_zscore > 0))
recall = with(subset(test, likert_rating_zscore > 0), mean(eharmony >= theta))
F = 2 * (precision * recall) / (precision + recall)
F

# Binary accuracy
test %>% mutate(
  accept = (eharmony >= theta),
  acc = ifelse(likert_rating_zscore > 0, accept, 1-accept)) ->
  test

test$likert_rating_binary <- ifelse(test$likert_rating_zscore >= 0, 1, 0)

test %>%
  summarise(acc = mean(acc))

test %>%
  group_by(likert_rating_binary) %>%
  summarise(acc = mean(acc))

test$likert_rating_binary <- ifelse(test$likert_rating_zscore > 0, 1, 0)
incorrect_predictions <- test %>%
  mutate(correct_prediction = ifelse(acc == 1, 1, 0)) %>%
  filter(correct_prediction == 0)

print(incorrect_predictions)




test$likert_rating_binary <- as.factor(test$likert_rating_binary)

ggplot(test, aes(x = harmony)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.5, colour = "black", fill = "white") +
  geom_density(alpha = 0.2, fill = "#FF6666") +
  geom_vline(aes(xintercept = theta), color = "blue", linetype = "dashed", size = 1) +
  geom_point(aes(y = likert_rating_binary, color = likert_rating_binary)) +
  theme_minimal() +
  labs(title = "Distribution of Harmony Scores",
       x = "Harmony Score",
       y = "Density",
       color = "likert_rating_binary") +
  scale_color_manual(values = c("red", "green"))

# Tuning percentage
# Function to calculate F score
calculate_F <- function(percent) {
  theta = quantile(subset(test, likert_rating_binary==1)$harmony, percent, na.rm=TRUE)
  
  precision = with(subset(test, harmony >= theta),
                   mean(likert_rating_binary == 1))
  recall = with(subset(test, likert_rating_binary==1),
                mean(harmony >= theta))
  F = 2 * (precision * recall) / (precision + recall)
  
  return(theta)
}


# Define the range of percent values
percent_values = seq(0, 1, 0.01)

# Calculate F score for each percent
F_values = sapply(percent_values, calculate_F)

# Find the percent that maximizes the F score
max_percent = percent_values[which.max(F_values)]

# Plot the relationship between percent and F score
plot(percent_values, F_values, type="b",
     xlab="Percent", ylab="F Score",
     main=paste("Maximum F Score at Percent =", round(max_percent, 2)))

# Sample data
test <- data.frame(
  harmony = c(0, 0, 1, 2, 3, 4, 5, 6, 7, 8),
  likert_rating = c(2, 4, 5, 3, 4, 5, 2, 3, 5, 4)
)


test <- data.frame(
  harmony = c(0, 99, 99, 99, 99, 99, 99, 99, 99, 99),
  likert_rating = c(6, 6, 6, 6, 6, 1, 1, 1, 1, 1)
)

# Calculate theta
theta <- with(subset(test, likert_rating > 3.5), quantile(-harmony, 0.99, na.rm = TRUE))

# Print theta
print(theta)



# Process HW learned models
# Load necessary libraries
library(dplyr)
library(stringr)
library(ggplot2)
library(readr)




# Loop over the directories
# Define the base directories
base_dirs <- c("/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/n=2", 
               "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/n=3")

# Create an empty final dataframe
final_df <- data.frame(stringsAsFactors = FALSE)
# Loop over the directories
for (base_dir in base_dirs) {
  # Get a list of all files in the current base directory
  all_files <- list.files(path = base_dir, pattern = "blickTestResults.txt", full.names = TRUE, recursive = TRUE)
  
  # Loop over all files
  for (file in all_files) {
    # Read the data into a dataframe
    df <- read.csv(file, sep = "\t", stringsAsFactors = FALSE, skip = 2, fill = TRUE, nrows = 0)
    df <- df[, 1:2]  # Keep only the first two columns
    
    # Rename the "score" column to "harmony"
    names(df)[names(df) == "score"] <- "harmony"
    
    # Split the file path into components
    path_components <- strsplit(file, split = "/")[[1]]
    
    # Extract 'n' from the path
    df$n <- gsub("n=", "", path_components[grep("n=", path_components)])
    
    # Extract 'maxoe' from the path, add a dot between the first and second digit
    maxoe <- path_components[length(path_components)-1]
    df$maxoe <- paste0(substr(maxoe, 1, 1), ".", substr(maxoe, 2, 2))
    
    # Append the dataframe to the final dataframe
    final_df <- rbind(final_df, df)
  }
}

# List all unique combinations of n and maxoe
combinations <- unique(final_df[, c("n", "maxoe")])
row.names(combinations) <- NULL

# Initialize an empty dataframe to store F scores
F_df <- data.frame()

# Loop over each unique combination of n and maxoe
for(i in seq_len(nrow(combinations))) {
  # Subset final_df for the current combination
  pred <- subset(final_df, n == combinations$n[i] & maxoe == combinations$maxoe[i])
  
  # Test data and predictions
  # test_path <- "path_to_your_test_data.csv" # <- Please replace this with your test data path
  test <- read.csv(test_path, sep=',', header = TRUE)
  

  # Join test and pred datasets
  test <- cbind(test, pred)
  
  test <- test %>%
    group_by(word) %>%
    summarise(
      harmony = mean(harmony),
      likert_rating = mean(likert_rating),
    ) %>%
    ungroup()
  
  # F-score
  theta = with(subset(test, likert_rating > 3.5), quantile(-harmony, 0.99, na.rm=TRUE))
  # how many harmony
  precision = with(subset(test, -harmony >= theta),
                   mean(likert_rating >  3.5))
  recall = with(subset(test, likert_rating >  3.5),
                mean(-harmony >= theta))
  F = 2 * (precision * recall) / (precision + recall)
  
  
  # Append the current n, maxoe, and F to F_df
  F_df <- rbind(F_df, data.frame(n = combinations$n[i], maxoe = combinations$maxoe[i], F = F))
}


# Set the theme
theme_set(theme_bw() +
            theme(
              text = element_text(family = "Times New Roman"),
              panel.grid.major = element_line(colour = "grey", linetype = "solid")
            )
)

# Create a labeller function
label_n <- function(variable,value){
  return(paste0(variable, " = ", value))
}

# Create the plot
p <- ggplot(F_df, aes(x = as.numeric(maxoe), y = F)) +
  geom_point(colour = "#3B76AF") +
  geom_line(colour = "#3B76AF") +
  facet_wrap(~n, labeller = label_n) +
  labs(x = "Threshold", y = "F-Score") +
  theme(plot.title = element_text(hjust = 0.5))

# Display the plot
print(p)

# Increase size of all text
p <- p + theme(text = element_text(size = 20))

# Save the plot
ggsave(filename = "MaxOE_F_Score.png", plot = p, dpi = 600, width = 10, height = 6)



# Save the plot
ggsave(filename = "MaxOE_F_Score.png", plot = p, dpi = 400)



# Testing

# Load necessary packages
library(gtools)
library(tidyverse)

# Assume observed_ratings is a vector of actual observed ratings
observed_ratings <- c(1, 2, 5, 5, 3, 1, 2, 2, 4, 5)

# Define bimodal distribution function
bimodal_dist <- function(x) {
  sapply(x, function(i) {
    if (i == 0) {
      sample(c(1, 2, 3), size = 1, prob = c(0.4, 0.3, 0.3))
    } else {
      sample(c(4, 5), size = 1, prob = c(0.3, 0.7))
    }
  })
}

# Define objective function
objective <- function(x) {
  generated_ratings <- bimodal_dist(x)
  return(mean((generated_ratings - observed_ratings)^2))
}

# Number of items
num_items <- length(observed_ratings)

# Generate all possible combinations of 0 and 1
combinations <- permutations(n = 2, r = num_items, v = c(0, 1), repeats.allowed = TRUE)

# Initialize dataframe to store results
results <- data.frame(combination = 1:nrow(combinations), objective_value = numeric())

# Loop through combinations
for (i in 1:nrow(combinations)) {
  # Calculate objective value
  obj_value <- objective(combinations[i,])
  
  # Store results
  results[i, "objective_value"] <- obj_value
}

# Print the combination with the smallest objective value
min_objective_value <- min(results$objective_value)
min_combination <- results$combination[which.min(results$objective_value)]
print(min_combination)





# Testing the likert rating of Fillers and nonce
# Read filler data
filler_file = "/Users/hutengdai/Documents/projects/exception-filtering-learner/data/polish/TestingData_filler.txt"
filler_data <- read.table(filler_file, header = FALSE, sep = "\t")

# Setting column names
names(filler_data) <- c("onset", "tail", "ucla", "subj", "likert", "token", "type")
filler_data$source <- "Filler"

# Read nonce data
nonce_file = "/Users/hutengdai/Documents/projects/exception-filtering-learner/data/polish/TestingData_novel.txt"
nonce_data <- read.table(nonce_file, header = FALSE, sep = "\t")

# Setting column names
names(nonce_data) <- c("onset", "tail", "ucla", "subj", "likert", "token", "type")
nonce_data$source <- "Nonce"

# Combine the data
combined_data <- rbind(filler_data, nonce_data)

# Calculate mean of Likert ratings by source and subject
aggregate(combined_data$likert, by = list(combined_data$source, combined_data$subj), FUN = mean, na.rm = TRUE)

