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
        cat("discordant:", x[i],x[j],y[i],y[j], "\n")
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

calculate_F <- function(percent) {
  theta = quantile(subset(test, likert_rating_binary==1)$harmony, percent, na.rm=TRUE)
  
  precision = with(subset(test, harmony >= theta),
                   mean(likert_rating_binary == 1))
  recall = with(subset(test, likert_rating_binary==1),
                mean(harmony >= theta))
  F = 2 * (precision * recall) / (precision + recall)
  
  return(theta)
}


setwd("~/Documents/projects/exception-filtering-learner")


# # english
# input_path <- "result/english/judgment_model-filtering_struc-local_flt-T_pad-F_conf-0.975_pen-10_thr-0.3.txt"
# input_path <- "result/english/judgment_model-gross_struc-local_flt-T_pad-F_conf-0.975_pen-10_thr-0.2.txt"
input_path <- "result/english/judgment_model-filtering_struc-local_flt-T_pad-F_conf-0.975_pen-10_thr-0.1.txt"
input_path <- "result/english/judgment_model-gross_struc-local_flt-T_pad-F_conf-0.975_pen-10_thr-0.1.txt"
input_path <- "result/english/judgment_model-gross_struc-local_flt-T_pad-F_conf-0.975_pen-10_thr-0.2.txt"
test_path <- "data/english/Daland_etal_2011_AverageScores.csv"
output_path <- "plot/english/correlation_plot.png"

test <- read.csv(test_path, sep=',', header = TRUE)
pred <- read.csv(input_path, sep='\t', header = FALSE)
colnames(pred) <- c("form","attestedness", "grammaticality")

test <- cbind(test, pred)
test$form <- trimws(test$form)

# Get the names of duplicate columns
duplicate_columns <- names(test)[duplicated(names(test))]
# Remove duplicate columns
test <- test[ , !duplicated(names(test))]

# ordinal regression
# install.packages("MASS")
# library(MASS)

# Fit the model
# test$likert_rating_order <- ordered(test$likert_rating, levels = c("1", "2", "3", "4", "5", "6"))  # adjust the levels according to your Likert scale
# mod_alt <- polr(likert_rating_order ~ grammaticality, data = test, Hess = TRUE)
# mod_null <- polr(likert_rating ~ 1, data = test, Hess=TRUE)

# Perform the likelihood ratio test
# library("lmtest")
# lrtest_result <- lrtest(mod_null, mod_alt)

# Print the result
# print(lrtest_result)

# Show summary of the model
# summary(mod_alt)
# Then you can compute AIC and BIC as usual
# aic <- AIC(mod_alt)
# bic <- BIC(mod_alt)

# print(paste("AIC: ", mod_alt))
# print(paste("BIC: ", mod_alt))

test <- test %>%
  group_by(form) %>%
  summarise(
    grammaticality = mean(grammaticality),
    likert_rating = mean(likert_rating),
    attestedness = first(attestedness)
    # likert_rating_zscore = mean(likert_rating_zscore)
  ) %>%
  ungroup()
# Test data and predictions
test$likert_rating_zscore <- scale(test$likert_rating)

test$likert_rating_binary <- ifelse(test$likert_rating_zscore >= 0, 1, 0)

correlation_methods <- c('spearman', 'kendall')
for (method in correlation_methods){
  s = cor(test$grammaticality, test$likert_rating, method=method)
  print(paste("NT: ", method, ": ", round(s, 3)))
}

gamma_result <- goodman_kruskals_gamma(test$likert_rating, test$grammaticality)
print(paste("Goodman Kruskal's Gamma: ", gamma_result))

# data_attested <- test[test$attestedness == "attested" | test$attestedness == "marginal", ]
data_attested <- test[test$attestedness == "attested" | test$attestedness == "marginal", ]
data_unattested <- test[test$attestedness == "unattested", ]
data_marginal <- test[test$attestedness == "marginal", ]


# Compute correlations for each group
for (method in correlation_methods) {
  # For attested data
  print(paste("Attested data correlations with method:", method))
  s = cor(data_attested$grammaticality, data_attested$likert_rating, method=method)
  print(paste("NT-: ", round(s, 3)))
  print(paste("Marginal data correlations with method:", method))
  s = cor(data_marginal$grammaticality, data_marginal$likert_rating, method=method)
  print(paste("NT: ", round(s, 3)))
  
  print(paste("Unattested data correlations with method:", method))
  s = cor(data_unattested$grammaticality, data_unattested$likert_rating, method=method)
  print(paste("NT: ", round(s, 3)))
  # s = cor(data_unattested$eharmony, data_unattested$likert_rating, method=method)
  # print(paste("UCLA: ", round(s, 3)))
}
gamma_result <- goodman_kruskals_gamma(data_attested$likert_rating, data_attested$grammaticality)
print(paste("Goodman Kruskal's Gamma: ", gamma_result))



#AIC BIC
# Fit the model
# model <- lmer(likert_rating ~ grammaticality, data = test)
# model <- lmer(likert_rating ~ grammaticality + (1|form), data = test)
model <- lm(likert_rating ~ grammaticality, data = test)
summary(model)


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

# F-score for Categorical

# Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
TP <- sum(test$grammaticality == 1 & test$likert_rating_zscore >= 0)
FP <- sum(test$grammaticality == 1 & test$likert_rating_zscore < 0)
FN <- sum(test$grammaticality == 0 & test$likert_rating_zscore >= 0)

# Calculate Precision, Recall
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

# Calculate F1 Score
F1 <- 2 * (precision * recall) / (precision + recall)

# Print the F1-score
print(F1)
print(precision)
print(recall)

# HW MaxEnt learner
# Test data and predictions
# maxent_path <- "result/english/EnglishJudgement_MaxEnt_OE03_unlimited.txt"
# maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/English/03OEunlimited-n=3/blickTestResults.txt"
# maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/English/03OEunlimited-n=2sigma1/blickTestResults.txt"
# maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/English/03OEunlimited-n=3sigma1/blickTestResults.txt"
maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/English/03OEunlimited-n=3sigma0/blickTestResults.txt"
maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/English/03OEunlimited-n=3sigma1-noCj/blickTestResults.txt"
# maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/English/03OEunlimited-n=2sigma01/blickTestResults.txt"


maxent_data <- read_delim(maxent_path, delim = '\t', skip = 2)

maxent_data <- maxent_data[, 1:2]  # Keep only the first two columns

test <- read.csv(test_path, sep=',', header = TRUE)
colnames(maxent_data) <- c('form',	'harmony')


test <- cbind(test, maxent_data)
test$eharmony <- eharmony_function(test$harmony)





# Fit the model
# library(ordinal)
# test$likert_rating <- ordered(test$likert_rating, levels = c("1", "2", "3", "4", "5", "6"))  # adjust the levels according to your Likert scale
# mod <- polr(likert_rating ~ eharmony, data = test, Hess = TRUE)
# mod_null <- polr(likert_rating ~ 1, data = test, Hess=TRUE)

# Perform the likelihood ratio test
# library("lmtest")
# lrtest_result <- lrtest(mod_null, mod)
# print(lrtest_result)

# Convert your likert_rating variable into an ordered factor

# Fit the model

# Show summary of the model
# summary(mod)
# Then you can compute AIC and BIC as usual
# aic <- AIC(mod)
# bic <- BIC(mod)

# print(paste("AIC: ", aic))
# print(paste("BIC: ", bic))


# scatter_plot <- ggplot(test, aes(x=grammaticality, y=likert_rating)) +
#   geom_smooth(method='lm', aes(x=grammaticality, y=likert_rating), color = 'gray') +
#   geom_point(aes(color=grammaticality, shape=grammaticality),show.legend=FALSE) + 
#   geom_text_repel(aes(label=form, color=grammaticality),  show.legend=FALSE, size=5) +  # Increase size here
#   scale_color_manual(values = c("dodgerblue3", "firebrick3")) +
#   labs(x='Predicted score', y='Proportion of "yes" responses') +
#   theme_classic(base_size = 20) +  # Increase size here
#   theme(
#     # legend.position=c(0.87, 0.15),
#     # legend.direction='vertical',
#     # legend.title=element_blank(),
#     text=element_text(family="Times New Roman", color="black", size=20)  # Increase size here
#   )+
#   scale_y_continuous(limits = c(0, 1))  # Set y-axis limits here
# 
# # Display the plot
# print(scatter_plot)
# 
# 
# # Save the plot as a high-resolution PNG
# ggsave("EF-zimmer-correlation.png", scatter_plot, dpi = 500, width = 6, height = 6)


# test <- test %>%
#   select(form, harmony, likert_rating)



test <- test %>%
  group_by(form) %>%
  summarise(
    harmony = mean(harmony),
    eharmony = mean(eharmony),
    likert_rating = mean(likert_rating),
    attestedness = first(attestedness)

  ) %>%
  ungroup()

# test$likert_rating_zscore <- scale(test$likert_rating)

correlation_methods <- c('pearson', 'spearman', 'kendall')
for (method in correlation_methods){
  s = cor(test$eharmony, test$likert_rating, method=method)
  
  print(paste("UCLA: ", method, ": ", round(s, 3)))
}

for (method in correlation_methods){
  test_result = cor.test(test$eharmony, test$likert_rating, method=method)
  s = test_result$estimate
  p = test_result$p.value
  
  print(paste("UCLA: ", method, ": correlation = ", round(s, 3), "p-value = ", p))
}
gamma_result <- goodman_kruskals_gamma(test$likert_rating, test$eharmony)
print(paste("Goodman Kruskal's Gamma: ", gamma_result))


data_attested <- test[test$attestedness == "attested" | test$attestedness == "marginal", ]
# data_attested <- test[test$attestedness == "attested", ]
data_unattested <- test[test$attestedness == "unattested", ]
data_marginal <- test[test$attestedness == "marginal", ]


# Compute correlations for each group
for (method in correlation_methods) {
  # For attested data
  print(paste("Attested data correlations with method:", method))
  s = cor(data_attested$eharmony, data_attested$likert_rating, method=method)
  print(paste("NT-: ", round(s, 3)))
  print(paste("Marginal data correlations with method:", method))
  s = cor(data_marginal$eharmony, data_marginal$likert_rating, method=method)
  print(paste("NT: ", round(s, 3)))
  
  print(paste("Unattested data correlations with method:", method))
  s = cor(data_unattested$eharmony, data_unattested$likert_rating, method=method)
  print(paste("NT: ", round(s, 3)))
  # s = cor(data_unattested$eharmony, data_unattested$likert_rating, method=method)
  # print(paste("UCLA: ", round(s, 3)))
}
gamma_result <- goodman_kruskals_gamma(data_attested$likert_rating, data_attested$eharmony)
print(paste("Goodman Kruskal's Gamma: ", gamma_result))




model <- lm(likert_rating ~ eharmony, data = test)
summary(model)

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
theta = optimize_theta(test)

# F-score
# theta = with(subset(test, likert_rating_zscore >= 0), quantile(eharmony, 0.99, na.rm=TRUE))
# Precision and recall
precision = with(subset(test, eharmony >= theta), mean(likert_rating_zscore >= 0))
recall = with(subset(test, likert_rating_zscore >= 0), mean(eharmony >= theta))
F = 2 * (precision * recall) / (precision + recall)
F
precision
recall



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
# theta <- with(subset(test, likert_rating > 3.5), quantile(-harmony, 0.99, na.rm = TRUE))

# Print theta
print(theta)

# Import the necessary library
library(ggplot2)
library(gridExtra)

# Create a data frame for the x values
df <- data.frame(x = seq(-400, 400, length.out = 1000))

# Create a function to generate each subplot
generate_plot <- function(sigma_value, xlab_show=FALSE) {
  p <- ggplot(df, aes(x)) +
    stat_function(fun = dnorm, args = list(mean = 0, sd = sigma_value), color = "blue") +
    labs(y = "Density", title = paste("σ =", sigma_value)) +
    theme(text = element_text(family = "Times New Roman"),
          plot.title = element_text(hjust = 0.5))  # Center the title
  
  if (xlab_show) {
    p <- p + xlab("Weight")
  } else {
    p <- p + theme(axis.title.x = element_blank())  # Hide x label
  }
  
  return(p)
}

# Generate the three subplots
p1 <- generate_plot(1,TRUE)
p2 <- generate_plot(10,TRUE)
p3 <- generate_plot(100, TRUE)  # Only show xlab in the last subplot

# Combine the subplots into a grid
plots <- arrangeGrob(p1, p2, p3, nrow = 1, top = "")
plots
ggsave(filename = "gassian-prior.png", plot = plots, dpi = 300)
