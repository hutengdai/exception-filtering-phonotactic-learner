require(tidyverse)

setwd("~/Documents/projects/exception-filtering-learner")


eharmony_function <- function(x){
  return(exp(-x))
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

input_path = "result/turkish/judgement_filtering_nonlocal_flt-T_pad-F_conf-0.975_pen-3_thr-0.5.txt"
# input_path = "result/turkish/judgement_filtering_nonlocal_flt-T_pad-F_conf-0.975_pen-10_thr-0.3.txt"

# input_path = "result/turkish/judgement_model-gross_struc-nonlocal_flt-T_pad-F_conf-0.975_pen-3_thr-0.5.txt"



# Read data from file
data <- read.table(input_path, sep = '\t', header = FALSE)

#1.categorical labels
# Set new column name for the last column
names(data)[1:2] <- c('form', 'grammaticality')
names(data)[ncol(data)] <- 'score'
data$boolean_grammaticality <- ifelse(data$grammaticality== "grammatical", 1, 0)
logisticModel <- glm(boolean_grammaticality ~ score, data = data, family = binomial)

summary(logisticModel)

#2. gradient
# Set new column names for the first two columns
names(data)[1:5] <- c('form', 'gradient','grammaticality', 'experiment', 'score')



pearsoncorr <- cor(data$score, data$gradient)
print(paste('Pearson correlation:', pearsoncorr))
spearmancorr <- cor(data$gradient, data$score, method="spearman")
print(paste('Spearman correlation:', spearmancorr))

kendalltau <- cor(data$gradient, data$score, method="kendall")
print(paste('Kendall type b correlation:', kendalltau))

gamma_result <- goodman_kruskals_gamma(data$gradient, data$score)
print(paste("Goodman Kruskal's Gamma: ", gamma_result)) 

#AIC BIC
# Fit the model
model <- glm(gradient ~ score, data = data)
shapiro.test(residuals(model))
qqnorm(residuals(model))
qqline(residuals(model))



# Assume you have fit a model
# model <- lm(judgment ~ binary, data = your_data)

# Calculate AIC
# AIC <- calculate_AIC(model, your_data$judgment)

summary(model)
# Now, you can compute AIC and BIC
aic <- AIC(model)
bic <- BIC(model)

# Print them out
print(paste("AIC: ", aic))
print(paste("BIC: ", bic))

# Fit the null model (intercept-only)
null_model <- lm(gradient ~ 1, data = data)

# Perform LRT
lrt_result <- anova(null_model, model)
print(lrt_result)


# Convert 'grammatical' to 1 and 'ungrammatical' to 0
data$judgment <- ifelse(data$grammaticality == 'grammatical', 1, 0)

# Calculate F1 score
precision <- sum((data$score == 1) & (data$judgment == 1)) / sum(data$score == 1)
recall <- sum((data$score == 1) & (data$judgment == 1)) / sum(data$judgment == 1)
f1 <- 2 * (precision * recall) / (precision + recall)

print(paste('Precision:', precision))
print(paste('Recall:', recall))
print(paste('F1 Score:', f1))

# Calculate overall accuracy
data$correct_prediction <- as.integer(data$judgment == data$score)
overall_accuracy <- mean(data$correct_prediction)
print(paste("Overall accuracy:", overall_accuracy))

# Calculate accuracy grouped by 'judgment'
grouped_accuracy <- tapply(data$correct_prediction, data$judgment, mean)
print(paste("Grouped accuracy:\n", grouped_accuracy))

# Filter and print incorrect predictions
incorrect_predictions <- subset(data, correct_prediction == 0)
print("Incorrect predictions")
print(incorrect_predictions)


data$form <- gsub(" ", "", data$form)
data$form <- gsub("a", "ɑ", data$form)
library(ggrepel)

scatter_plot <- ggplot(data, aes(x=score, y=gradient)) +
  geom_smooth(method='glm', aes(x=score, y=gradient), color = 'gray') +
  geom_point(aes(color=grammaticality, shape=grammaticality),show.legend=FALSE) + 
  geom_text_repel(aes(label=form, color=grammaticality),  show.legend=FALSE, size=5) +  # Increase size here
  scale_color_manual(values = c("dodgerblue3", "firebrick3")) +
  labs(x='Predicted score', y='Proportion of "yes" responses') +
  theme_classic(base_size = 20) +  # Increase size here
  theme(
    # legend.position=c(0.87, 0.15),
    # legend.direction='vertical',
    # legend.title=element_blank(),
    text=element_text(family="Times New Roman", color="black", size=20)  # Increase size here
  )+
  scale_y_continuous(limits = c(0, 1)) + # Set y-axis limits here
  scale_x_continuous(breaks = c(0,1))

# Display the plot
print(scatter_plot)


# Save the plot as a high-resolution PNG
ggsave("EF-zimmer-correlation.png", scatter_plot, dpi = 500, width = 6, height = 6)



# MaxEnt
data = read_tsv("data/turkish/TurkishTestingData.txt", 
                col_names = c('word', 'grammaticality'))
maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/Turkish/n=3/05OE/blickTestResults.txt"
maxent_path <- "/Users/hutengdai/Documents/projects/exception-filtering-learner/zimmer/07OE/blickTestResults-all-combs.txt"
maxent_path <- "/Users/hutengdai/Documents/projects/exception-filtering-learner/zimmer/07OE/blickTestResults-all-combs.txt"
maxent_path <- "/Users/hutengdai/Documents/projects/exception-filtering-learner/zimmer/07OEn=3con300sigma1-all/blickTestResults.txt"

maxent_data <- read_delim(maxent_path, delim = '\t', skip = 2)
maxent_data <- maxent_data[, 2]  # Keep only the first two columns

# Rename the "score" column to "harmony"
names(maxent_data)[names(maxent_data) == "score"] <- "harmony"

test <- cbind(data, maxent_data)


test %>% mutate(
  boolean_grammaticality = ifelse(grammaticality == 'grammatical', 1, 0)) ->
  test

test$eharmony <- sapply(test$harmony, eharmony_function)*1000

ggplot(test, aes(x=grammaticality, y=eharmony)) +
  geom_violin()


# find the best theta
optimize_theta <- function(data) {
  best_theta <- 0
  best_accuracy <- -Inf
  unique_values <- sort(unique(data$eharmony))

  for(theta in unique_values) {
    data$boolean_eharmony <- ifelse(data$eharmony >= theta, 1, 0)
    
    data %>% mutate(
      acc = ifelse(boolean_eharmony == boolean_grammaticality, 1, 0)) ->
      data
    
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


# Binary accuracy

test$boolean_eharmony <- ifelse(test$eharmony >= theta, 1, 0)

test %>% mutate(
  acc = ifelse(boolean_eharmony == boolean_grammaticality, 1, 0)) ->
  test

 test %>%
  summarise(acc = mean(acc))

test %>%
  group_by(grammaticality) %>%
  summarise(acc = mean(acc))

test %>%
  filter(grammaticality == 'ungrammatical') %>%
  arrange(eharmony) %>%
  head()


# Calculate overall accuracy
test$correct_prediction <- as.integer(test$boolean_eharmony == test$boolean_grammaticality)
overall_accuracy <- mean(test$correct_prediction)

# Calculate accuracy grouped by 'judgment'
grouped_accuracy <- tapply(test$correct_prediction, test$boolean_grammaticality, mean)

# Filter and print incorrect predictions
incorrect_predictions <- subset(test, correct_prediction == 0)
print(paste("Incorrect predictions:\n", incorrect_predictions))

# F-score
# theta = with(subset(test, grammaticality=='grammatical'),
#              quantile(eharmony, .99, na.rm=TRUE)
# )
precision = with(subset(test, eharmony >= theta),
                 mean(grammaticality == 'grammatical'))
recall = with(subset(test, grammaticality=='grammatical'),
              mean(eharmony >= theta))
F = 2 * (precision * recall) / (precision + recall)
F
precision
recall


logisticModel <- glm(boolean_grammaticality ~ eharmony, data = test, family = binomial)
summary(logisticModel)

data$predicted_probabilities <- predict(logisticModel, type = "response")
library(ggplot2)

ggplot(test, aes(x = eharmony/1000, y = boolean_grammaticality)) +
  geom_point() +
  geom_line(aes(y = boolean_grammaticality), colour = 'blue') +
  labs(y = "Predicted Probability", x = "Score") +
  theme_minimal()




pearsoncorr <- cor(test$boolean_grammaticality, test$eharmony)
print(paste('Pearson correlation:', pearsoncorr))

spearmancorr <- cor(test$boolean_grammaticality, test$eharmony, method="spearman")
print(paste('Spearman correlation:', spearmancorr))

kendalltau <- cor(test$boolean_grammaticality, test$eharmony, method="kendall")
print(paste('Kendall type b correlation:', kendalltau))

gamma_result <- goodman_kruskals_gamma(test$boolean_grammaticality, test$eharmony)
print(paste("Goodman Kruskal's Gamma: ", gamma_result)) 



# Heatmap
# Load the necessary libraries
library(dplyr)
library(tidyr)
library(ggplot2)

# The dataframe test is already defined here
# Define a pattern to extract vowels
pattern <- "[ieyɯuøao]"

# Apply the function to each word and create the new columns
test <- test %>%
  mutate(vowels = str_extract_all(word, pattern)) %>%
  mutate(
    first_vowel = sapply(vowels, function(x) x[1]),
    second_vowel = sapply(vowels, function(x) x[2])
  ) %>%
  select(-vowels) # remove the auxiliary column

# Define the ordering for y axis (first_vowel) and x axis (second_vowel)
vowel_order_y <- c("o", "u", "a", "ɯ", "ø", "y", "e", "i") # Reversed order for y axis
vowel_order_x <- c("i", "e", "y", "ø", "ɯ", "a", "u", "o") # Order for x axis

# Convert the first_vowel and second_vowel columns to ordered factors
test <- test %>%
  mutate(
    first_vowel = factor(first_vowel, levels = vowel_order_y),
    second_vowel = factor(second_vowel, levels = vowel_order_x)
  )

# Create the heatmap
heatmap_plot <- ggplot(test, aes(x=second_vowel, y=first_vowel, fill=eharmony)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(x="Second Vowel", y="First Vowel", fill="eHarmony", 
       title="Heatmap of eHarmony Values") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) # rotate x-axis labels

# backup data
data <- test

# To Latex
test <- data

library(dplyr)
library(tidyr)

# Select only necessary columns
test <- test %>% select(first_vowel, second_vowel, eharmony)

# Create the normalized harmony column
test$eharmony <- round((test$eharmony - min(test$eharmony)) / (max(test$eharmony) - min(test$eharmony)) * 100)
# test$eharmony <- round(test$eharmony / sum(test$eharmony) * 100)

# Reorder the first_vowel based on the vowel_order_x
test <- test %>% mutate(first_vowel = factor(first_vowel, levels = vowel_order_x)) %>%
  arrange(first_vowel)


# Convert data to wide format
wide_test <- pivot_wider(test, names_from = second_vowel, values_from = eharmony)

# Separate the data and labels for convenience
data_temp <- wide_test[-1]
labels <- wide_test[1]

# Convert the data_temp to numeric and replace NAs with 0
data_temp <- as.data.frame(lapply(data_temp, function(x) { as.numeric(ifelse(is.na(as.numeric(as.character(x))), 0, as.character(x))) }))

# Combine the data_temp and labels back together
matrix_test <- cbind(labels, data_temp)

# Convert wide format dataframe to matrix
matrix_test <- as.matrix(matrix_test)
rownames(matrix_test) <- matrix_test[,1]
matrix_test <- matrix_test[,-1]
# Reorder columns
matrix_test <- matrix_test[, vowel_order_x]

# Create a LaTeX table
latex_table <- "\\begin{tabular}{|c|" 
latex_table <- paste0(latex_table, paste(rep("c", ncol(matrix_test)), collapse = "|"), "|}\n\n")


for(i in 1:nrow(matrix_test)) {
  row_line <- paste0("", rownames(matrix_test)[i], " & ") 
  # Add each cell with cellcolor command
  for(j in 1:ncol(matrix_test)) {
    current_first_vowel <- rownames(matrix_test)[i]
    current_second_vowel <- colnames(matrix_test)[j]
    current_eharmony <- data$eharmony[data$first_vowel == current_first_vowel & data$second_vowel == current_second_vowel]
    if (is.na(matrix_test[i, j])) {
      row_line <- paste0(row_line, "\\cellcolor{myblue!0}")  # fill NA values with 0
    } else {
      row_line <- paste0(row_line, "\\cellcolor{myblue!", matrix_test[i, j], "}")
    }
    if (current_eharmony >= theta) {  # Checking if value >= theta
      row_line <- paste0(row_line, "\\color{white}1" , " & ")
    } else {
      row_line <- paste0(row_line, "\\color{black}0", "& ")
    }
  }
  row_line <- substr(row_line, 1, nchar(row_line)-2) # Remove last &
  row_line <- paste0(row_line, "\\\\ \n") # Add end of line
  latex_table <- paste0(latex_table, row_line) # Append row
}

latex_table <- paste0(latex_table, "\\end{tabular}\n")



# Print the LaTeX table
cat(latex_table)

# Zimmer's experiment

maxent_path <- "/Users/hutengdai/Desktop/UCLAPhonotacticLearner/WorkingFolder/Turkish/zimmer/05OEn=3/blickTestResults.txt"
maxent_path <- "/Users/hutengdai/Documents/projects/exception-filtering-learner/zimmer/07OE/response-rate/blickTestResults.txt"
maxent_path <- "/Users/hutengdai/Documents/projects/exception-filtering-learner/zimmer/07OE/blickTestResults.txt" # the best one

maxent_path <- "/Users/hutengdai/Documents/projects/exception-filtering-learner/zimmer/05OEn=3con100sigma1/blickTestResults.txt" # the best one
maxent_path <- "/Users/hutengdai/Documents/projects/exception-filtering-learner/zimmer/07OEn=3con300sigma1/blickTestResults.txt" # the best one

maxent_data <- read_delim(maxent_path, delim = '\t', skip = 2)
maxent_data <- maxent_data[, 2]  # Keep only the first two columns

# Rename the "score" column to "harmony"
names(maxent_data)[names(maxent_data) == "score"] <- "harmony"

data = read_tsv("data/turkish/TurkishTestingData-Zimmer.txt", 
                col_names = c('word', 'judgment', 'grammaticality', 'experiment'))

test <- cbind(data, maxent_data)

test$eharmony <- sapply(test$harmony, eharmony_function)



optimize_T <- function(test_data) {
  # Define a function that calculates negative Pearson correlation
  neg_pearson_corr <- function(T) {
    eharmony <- (test_data$eharmony)^(1/T)
    corr <- cor(test_data$judgment, eharmony)
    return(-corr)  # Minimize negative correlation to maximize positive correlation
  }
  
  # Optimize T to get the highest Pearson correlation
  result <- optimize(neg_pearson_corr, interval = c(0.01, 10))
  
  # Print optimized T and maximum correlation
  best_T <- result$minimum
  print(paste('Best T for maximum Pearson correlation:', best_T))
  print(paste('Maximum Pearson correlation:', -result$objective))
  
  return(best_T)
}

# Call the function with the test data
T <- optimize_T(test)
T
# Normalize test$eharmony
# test$eharmony <- test$eharmony^(1/T)

ggplot(test, aes(x=grammaticality, y=eharmony)) +
  geom_violin()

# Load the ggplot2 package
library(ggplot2)

# Calculate Pearson correlation
pearson_corr <- cor.test(test$eharmony, test$judgment, method="pearson")
annotation_pearman <- paste("Pearson: ", round(pearson_corr$estimate, 3))

library("ggrepel")

test$word <- gsub(" ", "", test$word)
test$word <- gsub("a", "ɑ", test$word)

test$harmonicity <- ifelse(test$grammaticality=="grammatical", "harmonic", "disharmonic")

scatter_plot <- ggplot(test, aes(x=eharmony, y=judgment)) +
  geom_smooth(method='glm', aes(x=eharmony, y=judgment), color = 'gray') +
  geom_point(aes(color=grammaticality, shape=grammaticality)) + 
  geom_text_repel(aes(label=word, color=grammaticality),  show.legend=FALSE, size=5) +  # Increase size here
  scale_color_manual(values = c("dodgerblue3", "firebrick3")) +
  labs(x=expression(exp(-harmony)), y='') +
  theme_classic(base_size = 20) +  # Increase size here
  theme(
    legend.position=c(0.85, 0.15),
    legend.direction='vertical',
    legend.title=element_blank(),
    text=element_text(family="Times New Roman", color="black", size=20)  # Increase size here
  )+
  scale_y_continuous(limits = c(0, 1))  # Set y-axis limits here

# Display the plot
print(scatter_plot)

# Save the plot as a high-resolution PNG
ggsave("HW-zimmer.png", scatter_plot, dpi = 500, width = 6, height = 6)




pearsoncorr <- cor(test$judgment, test$eharmony)
print(paste('Pearson correlation:', pearsoncorr))

spearmancorr <- cor(test$judgment, test$eharmony, method="spearman")
print(paste('Spearman correlation:', spearmancorr))

kendalltau <- cor(test$judgment, test$eharmony, method="kendall")
print(paste('Kendall type b correlation:', kendalltau))

gamma_result <- goodman_kruskals_gamma(test$judgment, test$eharmony)
print(paste("Goodman Kruskal's Gamma: ", gamma_result)) 

#AIC BIC
# Fit the model
model <- glm(judgment ~ eharmony, data = test)
summary(model)
shapiro.test(residuals(model))
qqnorm(residuals(model))
qqline(residuals(model))

# Now, you can compute AIC and BIC
aic <- AIC(model)
bic <- BIC(model)

# Print them out
print(paste("AIC: ", aic))
print(paste("BIC: ", bic))

# Fit the null model (intercept-only)
null_model <- lm(judgment ~ 1, data = test)

# Perform LRT
lrt_result <- anova(null_model, model)
print(lrt_result)



test %>% mutate(
  boolean_grammaticality = ifelse(grammaticality == 'grammatical', 1, 0)) ->
  test

optimize_theta <- function(data) {
  best_theta <- 0
  best_accuracy <- -Inf
  unique_values <- sort(unique(data$eharmony))
  
  for(theta in unique_values) {
    data$boolean_eharmony <- ifelse(data$eharmony >= theta, 1, 0)
    
    data %>% mutate(
      acc = ifelse(boolean_eharmony == boolean_grammaticality, 1, 0)) ->
      data
    
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

test$boolean_eharmony <- ifelse(test$eharmony >= theta, 1, 0)

# Calculate overall accuracy
test$correct_prediction <- as.integer(test$boolean_eharmony == test$boolean_grammaticality)
overall_accuracy <- mean(test$correct_prediction)
print(paste("Overall accuracy:", overall_accuracy))

# Calculate accuracy grouped by 'judgment'
grouped_accuracy <- tapply(test$correct_prediction, test$boolean_grammaticality, mean)
print(paste("Grouped accuracy:(ungrammatical then grammatical)", grouped_accuracy))

# Filter and print incorrect predictions
incorrect_predictions <- subset(test, correct_prediction == 0)
print(paste("Incorrect predictions:\n", incorrect_predictions))

# F-score
# theta = with(subset(test, grammaticality=='grammatical'),
#              quantile(eharmony, .99, na.rm=TRUE)
# )
precision = with(subset(test, eharmony >= theta),
                 mean(grammaticality == 'grammatical'))
recall = with(subset(test, grammaticality=='grammatical'),
              mean(eharmony >= theta))
F = 2 * (precision * recall) / (precision + recall)
F
precision
recall

