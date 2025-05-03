# Load Libraries
library(readr)
library(dplyr)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(syuzhet)
library(Matrix)
library(caret)
library(pROC)
library(e1071)
library(rpart)
library(ggplot2)
library(gridExtra)
library(xgboost)
library(caret)
library(Matrix)

# Load Data
train_data <- read_csv("data/EXIST2025_train.csv")
dev_data <- read_csv("data/EXIST2025_dev_labeled.csv")

# Exploratory Data Analysis
# Check dataset structure
cat("Train data dimensions:", dim(train_data), "\n")
cat("Distribution of classes in training data:\n")
# Rename the column for consistency
train_data$sexist <- tolower(train_data$label_task1_1)
print(table(train_data$sexist))
prop_sexist <- prop.table(table(train_data$sexist))
print(prop_sexist)

cat("Dev data dimensions:", dim(dev_data), "\n")
cat("Distribution of classes in dev data:\n")
# Ensure dev data also has the sexist column
if("label_task1_1" %in% colnames(dev_data)) {
  dev_data$sexist <- tolower(dev_data$label_task1_1)
} else if("sexist" %in% colnames(dev_data)) {
  # If already named correctly, make sure it's lowercase for consistency
  dev_data$sexist <- tolower(dev_data$sexist)
}
print(table(dev_data$sexist))

# Basic text analysis
# Create corpus
tweet_corpus <- corpus(train_data$tweet)

# Inspect the first few texts
cat("Sample tweets:\n")
head(as.character(tweet_corpus), 5)

# Create DFM with raw text
tokens_raw <- tokens(tweet_corpus)
dfm_raw <- dfm(tokens_raw)
print("Most frequent terms in raw text:")
print(topfeatures(dfm_raw, 20))

# Visualize raw text word cloud
png("raw_wordcloud.png", width=800, height=600)
textplot_wordcloud(dfm_raw, max_words = 100)
dev.off()

# Text Preprocessing
tokens_clean <- tokens(tweet_corpus,
                       remove_punct = TRUE,
                       remove_symbols = TRUE,
                       remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  tokens_trim()

dfm_clean <- dfm(tokens_clean)

# Word cloud with cleaned text
png("clean_wordcloud.png", width=800, height=600)
textplot_wordcloud(dfm_clean, max_words = 100)
dev.off()

# Most frequent terms (>200 occurrences)
top_terms <- topfeatures(dfm_clean, n = 200)
frequent_terms <- top_terms[top_terms > 200]
print("Terms occurring more than 200 times:")
print(frequent_terms)

# Collocations (words that appear together)
collocations <- textstat_collocations(tokens_clean, size = 2)
print("Top 10 collocations:")
print(head(collocations, 10))

# Co-occurrence with context window
fcmat <- fcm(tokens_clean, context = "window", window = 5)
top_features <- names(topfeatures(fcmat, 30))
fcmat_top <- fcm_select(fcmat, pattern = top_features)

# Visualize co-occurrence network
png("cooccurrence_network.png", width=1000, height=800)
textplot_network(fcmat_top, min_freq = 2, max_overlaps = 1000)
dev.off()

# Feature Engineering
# Add text length
train_data$tweet_length <- nchar(train_data$tweet)
dev_data$tweet_length <- nchar(dev_data$tweet)

# Compare tweet length by class
p1 <- ggplot(train_data, aes(x=sexist, y=tweet_length)) + 
  geom_boxplot() + 
  labs(title="Tweet Length by Sexist Label", x="Sexist", y="Character Count")

# Sentiment analysis
train_data$sentiment <- get_sentiment(train_data$tweet, method="syuzhet")
dev_data$sentiment <- get_sentiment(dev_data$tweet, method="syuzhet")

p2 <- ggplot(train_data, aes(x=sexist, y=sentiment)) + 
  geom_boxplot() + 
  labs(title="Sentiment Score by Sexist Label", x="Sexist", y="Sentiment Score")

# Emotion analysis with NRC lexicon
train_emotions <- get_nrc_sentiment(train_data$tweet)
train_data <- cbind(train_data, train_emotions)

dev_emotions <- get_nrc_sentiment(dev_data$tweet)
dev_data <- cbind(dev_data, dev_emotions)

# Compare emotions between sexist and non-sexist tweets
emotion_means <- train_data %>%
  group_by(sexist) %>%
  summarise(across(anger:trust, mean))

print("Emotion means by class:")
print(emotion_means)

# Visualization of emotions by class
emotion_data <- reshape2::melt(emotion_means, id.vars="sexist", 
                               variable.name="emotion", value.name="mean_score")

p3 <- ggplot(emotion_data, aes(x=emotion, y=mean_score, fill=sexist)) +
  geom_bar(stat="identity", position="dodge") +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  labs(title="Mean Emotion Scores by Class", x="Emotion", y="Mean Score")

# Arrange plots
grid.arrange(p1, p2, p3, ncol=2)

# Create document-feature matrix 
dfm_tfidf <- dfm_tfidf(dfm_clean)

# Handle duplicate tweets (same tweet annotated by multiple annotators)
# Group by tweet and take majority vote for the label
tweet_labels <- train_data %>%
  group_by(tweet) %>%
  summarize(sexist = ifelse(mean(sexist == "yes") >= 0.5, "yes", "no"))


# Create corpus from unique tweets
unique_corpus <- corpus(tweet_labels$tweet)

# Process unique tweets
unique_tokens <- tokens(unique_corpus,
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  tokens_trim()

unique_dfm <- dfm(unique_tokens)
unique_tfidf <- dfm_weight(unique_dfm)

# Convert to matrix format for modeling
X <- as.matrix(unique_tfidf)
y <- as.factor(tweet_labels$sexist)

# Create training and validation sets (80/20 split)
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex,]
X_test <- X[-trainIndex,]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Create a more manageable feature set (top 500 features by variance)
feat_variance <- apply(X_train, 2, var)
top_features <- names(sort(feat_variance, decreasing = TRUE))[1:500]
X_train_reduced <- X_train[, top_features]
X_test_reduced <- X_test[, top_features]

# IMPORTANT FIX: Extract emotion features from deduplicated tweets
# First, we need to merge the emotion data with our deduplicated tweets
# Create a mapping from tweet text to emotion features
emotion_mapping <- train_data %>%
  select(tweet, sentiment, anger, anticipation, disgust, fear, joy, 
         sadness, surprise, trust, negative, positive, tweet_length) %>%
  group_by(tweet) %>%
  summarize(across(c(sentiment, anger, anticipation, disgust, fear, joy, 
                     sadness, surprise, trust, negative, positive, tweet_length), mean))

# Now join this with our deduplicated tweet data
tweet_labels_with_emotions <- tweet_labels %>%
  left_join(emotion_mapping, by = "tweet")

# Now extract the emotion features for training and test sets
train_emotions <- as.matrix(tweet_labels_with_emotions[trainIndex, 
                                                       c("sentiment", "anger", "anticipation", 
                                                         "disgust", "fear", "joy", "sadness", 
                                                         "surprise", "trust", "negative", "positive",
                                                         "tweet_length")])
test_emotions <- as.matrix(tweet_labels_with_emotions[-trainIndex, 
                                                      c("sentiment", "anger", "anticipation", 
                                                        "disgust", "fear", "joy", "sadness", 
                                                        "surprise", "trust", "negative", "positive",
                                                        "tweet_length")])

# Now the dimensions should match and this should work without error
X_train_full <- cbind(X_train_reduced, train_emotions)
X_test_full <- cbind(X_test_reduced, test_emotions)

# Verify the dimensions match
cat("X_train_reduced dimensions:", dim(X_train_reduced), "\n")
cat("train_emotions dimensions:", dim(train_emotions), "\n")
cat("X_test_reduced dimensions:", dim(X_test_reduced), "\n")
cat("test_emotions dimensions:", dim(test_emotions), "\n")

# MODEL 1: Logistic Regression
cat("\nTraining Logistic Regression model...\n")
model_lr <- glm(y_train ~ ., data = as.data.frame(X_train_full), family = "binomial")
pred_lr <- predict(model_lr, as.data.frame(X_test_full), type = "response")
pred_class_lr <- ifelse(pred_lr > 0.5, "yes", "no")
conf_mat_lr <- confusionMatrix(as.factor(pred_class_lr), y_test)
roc_lr <- roc(as.numeric(y_test == "yes"), pred_lr)
auc_lr <- auc(roc_lr)

print("Logistic Regression Results:")
print(conf_mat_lr)
cat("AUC:", auc_lr, "\n")

# MODEL 2: Support Vector Machine
cat("\nTraining SVM model...\n")
model_svm <- svm(y_train ~ ., data = as.data.frame(X_train_full), probability = TRUE)
pred_svm <- predict(model_svm, as.data.frame(X_test_full), probability = TRUE)
pred_prob_svm <- attr(pred_svm, "probabilities")[,"yes"]
pred_class_svm <- ifelse(pred_prob_svm > 0.5, "yes", "no")
conf_mat_svm <- confusionMatrix(as.factor(pred_class_svm), y_test)
roc_svm <- roc(as.numeric(y_test == "yes"), pred_prob_svm)
auc_svm <- auc(roc_svm)

print("SVM Results:")
print(conf_mat_svm)
cat("AUC:", auc_svm, "\n")

# MODEL 3: Decision Tree
cat("\nTraining Decision Tree model...\n")
model_dt <- rpart(y_train ~ ., data = as.data.frame(X_train_full))
pred_dt <- predict(model_dt, as.data.frame(X_test_full), type = "prob")[,"yes"]
pred_class_dt <- ifelse(pred_dt > 0.5, "yes", "no")
conf_mat_dt <- confusionMatrix(as.factor(pred_class_dt), y_test)
roc_dt <- roc(as.numeric(y_test == "yes"), pred_dt)
auc_dt <- auc(roc_dt)

print("Decision Tree Results:")
print(conf_mat_dt)
cat("AUC:", auc_dt, "\n")

# MODEL 4: XGBOOST

y_train_num <- ifelse(y_train == "yes", 1, 0)
y_test_num <- ifelse(y_test == "yes", 1, 0)

X_train_xgb <- as.matrix(X_train_full)
X_test_xgb <- as.matrix(X_test_full)

# Create DMatrix objects for XGBoost
dtrain <- xgb.DMatrix(data = X_train_xgb, label = y_train_num)
dtest <- xgb.DMatrix(data = X_test_xgb, label = y_test_num)


cat("\nStarting hyperparameter optimisation with cross-validation...\n")

# List of parameters to test in cross-validation
param_grid <- list(
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1),
  gamma = c(0, 0.1, 0.3),
  subsample = c(0.7, 0.8, 0.9),
  colsample_bytree = c(0.7, 0.8, 0.9),
  min_child_weight = c(1, 3, 5)
)

# Function to evaluate a specific combination of parameters
evaluate_params <- function(max_depth, eta, gamma, subsample, colsample_bytree, min_child_weight) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = max_depth,
    eta = eta,
    gamma = gamma,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight
  )
  
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 100,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  # Return best AUC
  best_auc <- max(cv_results$evaluation_log$test_auc_mean)
  return(list(auc = best_auc, nrounds = which.max(cv_results$evaluation_log$test_auc_mean)))
}

# Initialise results
results <- data.frame()

# Limiting the number of combinations to save time
# Randomly selecting some combinations
set.seed(42)
max_depth_vals <- sample(param_grid$max_depth, 2)
eta_vals <- sample(param_grid$eta, 2)
gamma_vals <- sample(param_grid$gamma, 2)
subsample_vals <- sample(param_grid$subsample, 2)
colsample_vals <- sample(param_grid$colsample_bytree, 2)
min_child_vals <- sample(param_grid$min_child_weight, 2)

# Reduced search grid
for (depth in max_depth_vals) {
  for (eta_val in eta_vals) {
    for (gamma_val in gamma_vals) {
      for (subsample_val in subsample_vals) {
        for (colsample_val in colsample_vals) {
          for (min_child_val in min_child_vals) {
            cat("Evaluating: depth =", depth, "eta =", eta_val, "gamma =", gamma_val, "\n")
            
            result <- evaluate_params(depth, eta_val, gamma_val, subsample_val, colsample_val, min_child_val)
            
            results <- rbind(results, data.frame(
              max_depth = depth,
              eta = eta_val,
              gamma = gamma_val,
              subsample = subsample_val,
              colsample_bytree = colsample_val,
              min_child_weight = min_child_val,
              auc = result$auc,
              nrounds = result$nrounds
            ))
          }
        }
      }
    }
  }
}

# Find the best parameters
best_params_idx <- which.max(results$auc)
best_params <- results[best_params_idx, ]
cat("\nBest parameters found:\n")
print(best_params)

# Train the final model with the best parameters
final_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  gamma = best_params$gamma,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight
)

# Train the final model
xgb_final <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = best_params$nrounds,
  verbose = 0
)

cat("\nEvaluating XGBoost...\n")
pred_xgb <- predict(xgb_final, dtest)
pred_class_xgb <- ifelse(pred_xgb > 0.5, "yes", "no")
conf_mat_xgb <- confusionMatrix(as.factor(pred_class_xgb), y_test)
roc_xgb <- roc(as.numeric(y_test == "yes"), pred_xgb)
auc_xgb <- auc(roc_xgb)

# Display results
cat("\nXGBoost Results:\n")
print(conf_mat_xgb)
cat("AUC:", auc_xgb, "\n")


# Compare model performance including XGBoost
model_comparison <- data.frame(
  Model = c("Logistic Regression", "SVM", "Decision Tree", "XGBoost"),
  Accuracy = c(conf_mat_lr$overall["Accuracy"], 
               conf_mat_svm$overall["Accuracy"], 
               conf_mat_dt$overall["Accuracy"],
               conf_mat_xgb$overall["Accuracy"]),
  Precision = c(conf_mat_lr$byClass["Pos Pred Value"], 
                conf_mat_svm$byClass["Pos Pred Value"], 
                conf_mat_dt$byClass["Pos Pred Value"],
                conf_mat_xgb$byClass["Pos Pred Value"]),
  Recall = c(conf_mat_lr$byClass["Sensitivity"], 
             conf_mat_svm$byClass["Sensitivity"], 
             conf_mat_dt$byClass["Sensitivity"],
             conf_mat_xgb$byClass["Sensitivity"]),
  F1_Score = c(conf_mat_lr$byClass["F1"], 
               conf_mat_svm$byClass["F1"], 
               conf_mat_dt$byClass["F1"],
               conf_mat_xgb$byClass["F1"]),
  AUC = c(auc_lr, auc_svm, auc_dt, auc_xgb)
)


print("Model Performance Comparison:")
print(model_comparison)

# Plot ROC curves for all models
png("roc_curves.png", width=800, height=600)
plot(roc_lr, col = "blue", main = "ROC Curve Comparison")
plot(roc_svm, col = "red", add = TRUE)
plot(roc_dt, col = "green", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "SVM", "Decision Tree"),
       col = c("blue", "red", "green"), lwd = 2)
dev.off()

