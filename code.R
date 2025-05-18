

# Task 1: Sexism Identification Classifier


# Libraries

library(quanteda)
library(quanteda.textmodels)
library(quanteda.textplots)
library(quanteda.textstats)
library(readr)
library(dplyr)
library(recommenderlab)
library(syuzhet)
library(textclean)
library(stringr)
library(tidyverse)
library(gridExtra)
library(corrplot)
library(purrr)
library(Matrix)
library(caret)
library(pROC)
library(e1071)
library(rpart)
library(ggplot2)
library(xgboost)
library(themis)
library(recipes)
library(glmnet)
library(randomForest)
library(knitr)        
library(ggcorrplot)   
library(scales)      
library(viridis)      
library(tidytext)     
library(ggwordcloud) 
library(pdp)   
library(parallel)
library(doParallel)
library(rpart.plot)

theme_set(theme_minimal(base_size = 12) + 
            theme(
              plot.title = element_text(face = "bold", size = 14),
              plot.subtitle = element_text(size = 11),
              axis.title = element_text(face = "bold"),
              legend.title = element_text(face = "bold"),
              panel.grid.minor = element_blank(),
              panel.border = element_rect(color = "gray80", fill = NA)
            ))

# Custom colors for consistent visualization
sexist_colors <- c("no" = "#3498db", "yes" = "#e74c3c")

# Data Loading

train_data <- read_csv("data/EXIST2025_train.csv")
dev_data <- read_csv("data/EXIST2025_dev_labeled.csv")

# Display dataset information
message("Dataset Dimensions:")
data_dims <- data.frame(
  Dataset = c("Training", "Development"),
  Rows = c(nrow(train_data), nrow(dev_data)),
  Columns = c(ncol(train_data), ncol(dev_data))
)
kable(data_dims)


# Ensure consistent label naming
train_data$sexist <- tolower(train_data$label_task1_1)
if("label_task1_1" %in% colnames(dev_data)) {
  dev_data$sexist <- tolower(dev_data$label_task1_1)
} else if("sexist" %in% colnames(dev_data)) {
  dev_data$sexist <- tolower(dev_data$sexist)
}

# Label distribution visualization
train_labels <- table(train_data$sexist)
train_prop <- prop.table(train_labels)

# Create label distribution plot
ggplot(data.frame(label = names(train_labels), count = as.numeric(train_labels)), 
       aes(x = label, y = count, fill = label)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = count), vjust = -0.5, size = 4) +
  geom_text(aes(label = percent(count/sum(count), accuracy = 0.1)), vjust = 1.5, color = "white", size = 4) +
  labs(title = "Distribution of Classes in Training Data",
       subtitle = "Count and percentage of sexist vs. non-sexist tweets",
       x = "Classification", y = "Count") +
  scale_fill_manual(values = sexist_colors) +
  theme(legend.position = "none")


# Grouping by tweet (handling multiple annotations)

# Aggregate multiple annotations with majority vote
train_grouped <- train_data %>%
  group_by(tweet) %>%
  summarise(
    count_yes = sum(sexist == "yes", na.rm = TRUE),
    count_no = sum(sexist == "no", na.rm = TRUE),
    annotator_count = n()
  ) %>%
  filter(count_yes != count_no) %>% # Remove ties
  mutate(
    final_label = ifelse(count_yes > count_no, "yes", "no"),
    agreement_ratio = pmax(count_yes, count_no) / annotator_count
  )

# Analyze annotation agreement
agreement_summary <- train_grouped %>%
  group_by(final_label) %>%
  summarise(
    n = n(),
    mean_agreement = mean(agreement_ratio),
    median_agreement = median(agreement_ratio)
  )

# Visualize annotator agreement by class
ggplot(train_grouped, aes(x = agreement_ratio, fill = final_label)) +
  geom_histogram(binwidth = 0.05, position = "dodge", alpha = 0.8) +
  labs(title = "Annotator Agreement by Class",
       subtitle = "Higher values indicate stronger consensus among annotators",
       x = "Agreement Ratio", y = "Count", fill = "Classification") +
  scale_fill_manual(values = sexist_colors) +
  facet_wrap(~final_label, scales = "free_y") +
  scale_x_continuous(limits = c(0.5, 1), breaks = seq(0.5, 1, 0.1))

# Display label distribution after grouping
final_label_dist <- table(train_grouped$final_label)
kable(data.frame(
  Class = names(final_label_dist),
  Count = as.numeric(final_label_dist),
  Percentage = percent(as.numeric(final_label_dist) / sum(final_label_dist))
), caption = "Final Label Distribution After Grouping")


# Text Pre-processing Functions

# Function for cleaning, tokenizing, removing stop words, numbers, and applying stemming
process_tweets <- function(tweets) {
  # Initial cleaning
  tweets <- replace_html(tweets)                             # expose ampersands (&) to be removed later
  tweets <- str_replace_all(tweets, "@\\w+", "USER")         # replace usernames with USER token
  tweets <- str_replace_all(tweets, "http\\S+", "URL")       # replace links with URL token
  tweets <- str_replace_all(tweets, "\\.(\\S)", ". \\1")     # add space after dots
  # Create corpus
  corpus <- corpus(tweets)
  # Tokenize with careful handling of important features
  toks <- tokens(
    corpus,
    remove_punct = TRUE,
    remove_symbols = TRUE,
    what = "word"
  )
  # Normalize text
  toks <- tokens_tolower(toks)
  toks <- tokens_remove(toks, stopwords("english"))
  toks <- tokens_remove(toks, pattern = "^\\d+$", valuetype = "regex")
  toks <- tokens_wordstem(toks)
  
  return(toks)
}

# Function to extract n-grams from text
extract_ngrams <- function(text, n = 2) {
  toks <- tokens(text)
  toks_ngrams <- tokens_ngrams(toks, n = n)
  return(toks_ngrams)
}

# Text Processing and Initial Analysis

# Create a corpus for analysis and visualization
tweet_corpus <- corpus(train_grouped$tweet)

# Sample of tweets
cat("Sample tweets:\n")
head(as.character(tweet_corpus), 3)

# Process tweets
toks <- process_tweets(train_grouped$tweet)
dfm <- dfm(toks)
dfm_tfidf <- dfm_tfidf(dfm)

# Extract top features from the dfm and assign them properly
top_features <- topfeatures(dfm, 20)

# Now this will work
top_features_df <- data.frame(
  word = names(top_features),
  frequency = as.numeric(top_features)
)

ggplot(top_features_df, aes(x = reorder(word, frequency), y = frequency)) +
  geom_col(fill = "#1f77b4") +
  coord_flip() +
  labs(
    title = "Most Frequent Terms in Processed Text",
    subtitle = "After removing stopwords and stemming",
    x = "Term", y = "Frequency"
  )

ggplot(top_features_df[1:100,], aes(label = word, size = frequency, color = frequency)) +
  geom_text_wordcloud_area() +
  scale_size_area(max_size = 15) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "Word Cloud of Most Frequent Terms")

# Feature Engineering: Important Words

# Analyze important words for each class
dfm_yes <- dfm_subset(dfm_tfidf, train_grouped$final_label == "yes")
dfm_no <- dfm_subset(dfm_tfidf, train_grouped$final_label == "no")

top_yes <- topfeatures(dfm_yes, 30)
top_no <- topfeatures(dfm_no, 30)

cat("Top words in 'YES' tweets:\n")
print(top_yes)
cat("\nTop words in 'NO' tweets:\n")
print(top_no)

# Count documents in each class
n_yes <- ndoc(dfm_yes)
n_no <- ndoc(dfm_no)

# Calculate relative frequencies with keyness statistics
keyness <- textstat_keyness(dfm, train_grouped$final_label == "yes")
keyness_df <- data.frame(keyness)
keyness_df$significant <- ifelse(keyness_df$p < 0.05, "Yes", "No")

# Class-specific words with relative frequencies
df_yes <- data.frame(word = names(top_yes), count = as.numeric(top_yes)) %>%
  mutate(relative_count = count / n_yes,
         class = "Sexist") %>%
  arrange(desc(relative_count))

df_no <- data.frame(word = names(top_no), count = as.numeric(top_no)) %>%
  mutate(relative_count = count / n_no,
         class = "Non-sexist") %>%
  arrange(desc(relative_count))

# Combined dataframe for visualization
class_words <- rbind(df_yes[1:15,], df_no[1:15,])

# Visualize class-specific words with better formatting
ggplot(class_words, aes(x = reorder(word, relative_count), y = relative_count, fill = class)) +
  geom_col() +
  facet_wrap(~ class, scales = "free_y") +
  coord_flip() +
  scale_fill_manual(values = c("Sexist" = "#e74c3c", "Non-sexist" = "#3498db")) +
  labs(title = "Most Characteristic Words by Class",
       subtitle = "TF-IDF weighted relative frequencies",
       x = "Word", y = "Relative Frequency") +
  theme(legend.position = "none")

# Visualize keyness statistics
ggplot(head(keyness_df, 20), aes(x = reorder(feature, chi2), y = chi2, fill = chi2 > 0)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#e74c3c", "FALSE" = "#3498db"), 
                    labels = c("Associated with sexist", "Associated with non-sexist")) +
  labs(title = "Keyness Statistics: Words Associated with Classes",
       subtitle = "Chi-squared values indicating word-class associations",
       x = "Word", y = "Chi-squared value", fill = "Association") +
  theme(legend.position = "bottom")

# Function to create binary features for important words
important_words <- function(df, tweet_col = "tweet") {
  # Expanded list of important words identified from keyness analysis
  words <- c("woman", "women", "men", "girl", "sex", "bitch", "fuck",
             "love", "peopl", "gender", "femal", "male", "stupid", 
             "kitchen", "joke", "victim", "harass", "equal", "feminist")
  
  df_copy <- df
  tweets_lower <- tolower(df_copy[[tweet_col]])
  
  for (word in words) {
    col_name <- word
    # Replace fixed() with a regular expression pattern
    df_copy[[col_name]] <- as.integer(str_detect(tweets_lower, paste0("\\b", word, "\\b")))
  }
  
  return(df_copy)
}

important_words <- function(df, tweet_col = "tweet") {
  # Expanded list of important words identified from keyness analysis
  words <- c("woman", "women", "men", "girl", "sex", "bitch", "fuck",
             "love", "peopl", "gender", "femal", "male", "stupid", 
             "kitchen", "joke", "victim", "harass", "equal", "feminist")
  
  df_copy <- df
  tweets_lower <- tolower(df_copy[[tweet_col]])
  
  for (word in words) {
    col_name <- word
    # Replace fixed() with a regular expression pattern
    df_copy[[col_name]] <- as.integer(str_detect(tweets_lower, paste0("\\b", word, "\\b")))
  }
  
  return(df_copy)
}

# Function to analyze context of key words
analyze_context <- function(tokens, target_word, window = 3) {
  tokens_context <- tokens_select(tokens, pattern = target_word, selection = "keep", 
                                  window = window)
  dfm_context <- dfm(tokens_context)
  context_features <- topfeatures(dfm_context, 10)
  return(context_features)
}

# Analyze collocations (word pairs) for each class
tweets_yes <- train_grouped %>% filter(final_label == "yes") %>% pull(tweet)
tweets_no <- train_grouped %>% filter(final_label == "no") %>% pull(tweet)

toks_yes <- process_tweets(tweets_yes)
toks_no <- process_tweets(tweets_no)

# Analyze bigrams (more interpretable than collocations)
dfm_yes_bigrams <- dfm(tokens_ngrams(toks_yes, n = 2))
dfm_no_bigrams <- dfm(tokens_ngrams(toks_no, n = 2))

top_bigrams_yes <- topfeatures(dfm_yes_bigrams, 20)
top_bigrams_no <- topfeatures(dfm_no_bigrams, 20)

# Create data frames for visualization
bigrams_yes_df <- data.frame(
  bigram = names(top_bigrams_yes),
  count = as.numeric(top_bigrams_yes),
  class = "Sexist"
) %>% arrange(desc(count))

bigrams_no_df <- data.frame(
  bigram = names(top_bigrams_no),
  count = as.numeric(top_bigrams_no),
  class = "Non-sexist"
) %>% arrange(desc(count))

# Combine data frames
bigrams_df <- rbind(
  head(bigrams_yes_df, 10),
  head(bigrams_no_df, 10)
)

# Visualize top bigrams by class
ggplot(bigrams_df, aes(x = reorder(bigram, count), y = count, fill = class)) +
  geom_col() +
  facet_wrap(~ class, scales = "free_y") +
  coord_flip() +
  scale_fill_manual(values = c("Sexist" = "#e74c3c", "Non-sexist" = "#3498db")) +
  labs(title = "Most Common Bigrams by Class",
       subtitle = "Frequency of word pairs in each class",
       x = "Bigram", y = "Count") +
  theme(legend.position = "none")

# Function to create enhanced collocation features
coloc <- function(df) {
  # Extract most significant collocations for each class
  collocs_yes <- textstat_collocations(toks_yes, size = 2, min_count = 5)
  collocs_no <- textstat_collocations(toks_no, size = 2, min_count = 5)
  
  # Get top collocations by effect size (higher lambda means stronger association)
  collocations_yes <- collocs_yes %>% 
    arrange(desc(lambda)) %>% 
    head(15) %>% 
    pull(collocation)
  
  collocations_no <- collocs_no %>% 
    arrange(desc(lambda)) %>% 
    head(15) %>% 
    pull(collocation)
  
  # Create binary features for each top collocation
  df$sexist_colloc_count <- 0
  df$non_sexist_colloc_count <- 0
  
  for (colloc in collocations_yes) {
    col_name <- paste0("yes_", gsub(" ", "_", colloc))
    
    df[[col_name]] <- as.integer(str_detect(tolower(df$tweet), colloc))  
    df$sexist_colloc_count <- df$sexist_colloc_count + df[[col_name]]
  }
  
  for (colloc in collocations_no) {
    col_name <- paste0("no_", gsub(" ", "_", colloc))
    
    df[[col_name]] <- as.integer(str_detect(tolower(df$tweet), colloc)) 
    df$non_sexist_colloc_count <- df$non_sexist_colloc_count + df[[col_name]]
  }
  
  return(df)
}


# Feature Engineering: Sentiment and Emotion Analysis

# Function to analyze sentiment sequences
sent_seq <- function(data) {
  data$tweet <- as.character(data$tweet)
  sentiment_results <- data.frame(tweet = data$tweet)
  # Extract sentences
  sentences_per_tweet <- lapply(data$tweet, get_sentences)
  # Calculate overall sentiment
  sentiment_results$overall_sentiment <- get_sentiment(data$tweet, method = "syuzhet")
  # Calculate sentence-level sentiment
  sentiments_per_tweet <- lapply(sentences_per_tweet, get_sentiment, method = "syuzhet")
  # Calculate sentiment patterns
  sentiment_results$sentiment_variance <- sapply(sentiments_per_tweet, function(s) {
    if (length(s) > 1) var(s) else 0
  })
  sentiment_results$sentiment_range <- sapply(sentiments_per_tweet, function(s) {
    if (length(s) > 1) max(s) - min(s) else 0
  })
  sentiment_results$sentiment_shift <- sapply(sentiments_per_tweet, function(s) {
    if (length(s) > 1) s[length(s)] - s[1] else 0
  })
  sentiment_results$all_pos <- sapply(sentiments_per_tweet, function(sentiments) {
    if (length(sentiments) > 0 && all(sign(sentiments) == 1)) 1 else 0
  })
  sentiment_results$all_neg <- sapply(sentiments_per_tweet, function(sentiments) {
    if (length(sentiments) > 0 && all(sign(sentiments) == -1)) 1 else 0
  })
  sentiment_results$mixed_sentiment <- sapply(sentiments_per_tweet, function(sentiments) {
    if (length(sentiments) > 1 && any(sign(sentiments) == 1) && any(sign(sentiments) == -1)) 1 else 0
  })
  # Return only the calculated features (not the tweet text)
  return(sentiment_results[, -1])
}

# Enhanced function for emotional features
extract_emotions <- function(texts) {
  # Get NRC emotions
  emotions <- get_nrc_sentiment(texts)
  
  # Define expected emotion columns
  emotion_cols <- c("anger", "anticipation", "disgust", "fear", 
                    "joy", "sadness", "surprise", "trust", 
                    "positive", "negative")
  
  # Add any missing columns with 0s
  for (col in emotion_cols) {
    if (!(col %in% colnames(emotions))) {
      emotions[[col]] <- 0
    }
  }
  
  # Rename emotion columns to include _nrc suffix
  rename_map <- setNames(paste0(emotion_cols, "_nrc"), emotion_cols)
  names(emotions)[names(emotions) %in% names(rename_map)] <- rename_map[names(emotions)[names(emotions) %in% names(rename_map)]]
  
  # Calculate emotion ratios
  emotions$neg_to_pos_ratio <- ifelse(emotions$positive_nrc > 0, 
                                      emotions$negative_nrc / emotions$positive_nrc, 
                                      emotions$negative_nrc)
  
  # Emotional complexity
  basic_emotions <- paste0(c("anger", "anticipation", "disgust", "fear", 
                             "joy", "sadness", "surprise", "trust"), "_nrc")
  emotions$emotion_complexity <- apply(emotions[, basic_emotions], 1, function(row) {
    sum(row > 0)
  })
  
  # Dominant emotion
  emotions$dominant_emotion <- apply(emotions[, basic_emotions], 1, function(row) {
    if (all(row == 0)) return("none")
    c("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")[which.max(row)]
  })
  
  return(emotions)
}

# Function to calculate linguistic complexity features
linguistic_complexity <- function(texts) {
  # Initialize results data frame
  results <- data.frame(
    avg_word_length = numeric(length(texts)),
    avg_sentence_length = numeric(length(texts)),
    lexical_diversity = numeric(length(texts)),
    question_marks = numeric(length(texts)),
    exclamation_marks = numeric(length(texts)),
    capitalized_ratio = numeric(length(texts))
  )
  
  for (i in seq_along(texts)) {
    text <- texts[i]
    
    # Words
    words <- unlist(str_split(text, "\\s+"))
    words <- words[words != ""]
    
    # Average word length
    results$avg_word_length[i] <- mean(nchar(words), na.rm = TRUE)
    
    # Sentences
    sentences <- get_sentences(text)
    
    # Average sentence length
    results$avg_sentence_length[i] <- mean(sapply(sentences, function(s) {
      length(unlist(str_split(s, "\\s+")))
    }), na.rm = TRUE)
    
    # Lexical diversity (unique words / total words)
    if (length(words) > 0) {
      results$lexical_diversity[i] <- length(unique(words)) / length(words)
    } else {
      results$lexical_diversity[i] <- 0
    }
    
    # Question marks
    results$question_marks[i] <- str_count(text, "\\?")
    
    # Exclamation marks
    results$exclamation_marks[i] <- str_count(text, "!")
    
    # Capitalization ratio
    cap_chars <- sum(str_count(text, "[A-Z]"))
    all_chars <- sum(str_count(text, "[a-zA-Z]"))
    results$capitalized_ratio[i] <- ifelse(all_chars > 0, cap_chars / all_chars, 0)
  }
  
  return(results)
}

# Comprehensive Feature Set Construction

# Add basic word features
train_grouped <- important_words(train_grouped)
dev_data <- important_words(dev_data)

# Add collocation features
train_grouped <- coloc(train_grouped)
dev_data <- coloc(dev_data)

# Add textual statistics
train_grouped$tweet_length <- nchar(train_grouped$tweet)
train_grouped$word_count <- sapply(strsplit(train_grouped$tweet, "\\s+"), length)

dev_data$tweet_length <- nchar(dev_data$tweet)
dev_data$word_count <- sapply(strsplit(dev_data$tweet, "\\s+"), length)

# Add linguistic complexity features
train_ling <- linguistic_complexity(train_grouped$tweet)
dev_ling <- linguistic_complexity(dev_data$tweet)

train_grouped <- cbind(train_grouped, train_ling)
dev_data <- cbind(dev_data, dev_ling)

# Add sentiment sequence features
train_sent <- sent_seq(train_grouped)
dev_sent <- sent_seq(dev_data)

train_grouped <- cbind(train_grouped, train_sent)
dev_data <- cbind(dev_data, dev_sent)

# Add emotion features
train_emotions <- extract_emotions(train_grouped$tweet)
dev_emotions <- extract_emotions(dev_data$tweet)

train_grouped <- cbind(train_grouped, train_emotions)
dev_data <- cbind(dev_data, dev_emotions)

# Feature Analysis and Selection

# Compare emotions between sexist and non-sexist tweets
# Visualize emotion distribution by class
emotion_means <- train_grouped %>%
  group_by(final_label) %>%
  summarise(across(c(anger_nrc, anticipation_nrc, disgust_nrc, fear_nrc, joy_nrc, sadness_nrc, surprise_nrc, trust_nrc, negative_nrc, positive_nrc), mean, na.rm = TRUE))

# Reshaping the data for visualization
emotion_data <- reshape2::melt(emotion_means, id.vars="final_label", 
                               variable.name="emotion", value.name="mean_score")

# Enhanced emotion visualization
ggplot(emotion_data, aes(x = emotion, y = mean_score, fill = final_label)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = sexist_colors, name = "Class") +
  labs(title = "Mean Emotion Scores by Class",
       subtitle = "Emotions detected in sexist vs. non-sexist tweets",
       x = "Emotion", y = "Mean Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Examine dominant emotions
dominant_counts <- train_grouped %>%
  group_by(final_label, dominant_emotion) %>%
  summarise(count = n()) %>%
  group_by(final_label) %>%
  mutate(proportion = count / sum(count))

# Visualize dominant emotions by class
ggplot(dominant_counts, aes(x = dominant_emotion, y = proportion, fill = final_label)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = sexist_colors, name = "Class") +
  labs(title = "Dominant Emotions by Class",
       subtitle = "Proportion of tweets with each dominant emotion",
       x = "Dominant Emotion", y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Analyze linguistic complexity features by class
ling_means <- train_grouped %>%
  group_by(final_label) %>%
  summarise(across(c(avg_word_length, avg_sentence_length, lexical_diversity,
                     question_marks, exclamation_marks, capitalized_ratio), 
                   mean, na.rm = TRUE))

ling_data <- reshape2::melt(ling_means, id.vars="final_label", 
                            variable.name="feature", value.name="mean_value")

# Visualize linguistic complexity by class
ggplot(ling_data, aes(x = feature, y = mean_value, fill = final_label)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = sexist_colors, name = "Class") +
  labs(title = "Linguistic Complexity Features by Class",
       subtitle = "Mean values of text complexity metrics",
       x = "Feature", y = "Mean Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Calculate feature correlations with target
model_features <- train_grouped %>%
  select(-tweet, -count_yes, -count_no, -annotator_count, -agreement_ratio)

# Convert target to numeric
model_features$target_numeric <- as.numeric(model_features$final_label == "yes")

# Calculate point-biserial correlation for each feature with the target
correlations <- sapply(select_if(model_features, is.numeric), function(x) {
  if(var(x, na.rm = TRUE) > 0) {
    cor(x, model_features$target_numeric, use = "pairwise.complete.obs")
  } else {
    0
  }
})

# Filter out NAs
correlations <- correlations[!is.na(correlations)]

# Create a data frame for visualization
cor_df <- data.frame(
  feature = names(correlations),
  correlation = as.numeric(correlations)
) %>%
  arrange(desc(abs(correlation)))

# Visualize top correlated features
ggplot(head(cor_df, 25), aes(x = reorder(feature, abs(correlation)), y = correlation, fill = correlation > 0)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#e74c3c", "FALSE" = "#3498db"), 
                    labels = c("Positive correlation", "Negative correlation")) +
  labs(title = "Features Most Correlated with Sexism",
       subtitle = "Point-biserial correlation between features and target",
       x = "Feature", y = "Correlation Coefficient", fill = "Direction") +
  theme(legend.position = "bottom")

head(cor_df, 50)

# Feature Selection

# Select features for modeling based on analysis
features <- c(
  # === Palavra-chave sobre gênero ou insultos ===
  "women", "men", "woman", "bitch", "fuck", "sex", "girl", "male", "love", "feminist",
  
  # === Expressões específicas ou collocations ===
  "sexist_colloc_count", "non_sexist_colloc_count",
  "yes_dumb_blond", "yes_short_skirt", "yes_cock_carousel",
  "yes_gold_digger", "no_gold_digger", "yes_blah_blah", 
  "no_victim_card", "no_knee_slapper", "no_virginia_woolf",
  "yes_everyday_sexism", "no_everyday_sexism",
  "yes_cock_teas", "no_cock_teas", "no_mental_ill",
  "no_sexual_orient", "no_voter_suppress",
  
  # === Sentimento geral e complexidade emocional ===
  "overall_sentiment", "all_pos", "all_neg", "mixed_sentiment",
  "neg_to_pos_ratio", "emotion_complexity", "sentiment_range",
  
  # === Emoções específicas (NRC lexicon) ===
  "disgust_nrc", "sadness_nrc", "negative_nrc", "positive_nrc", 
  "trust_nrc", "anger_nrc", "fear_nrc", "joy_nrc",
  
  # === Complexidade linguística ===
  "avg_word_length", "question_marks", "capitalized_ratio", 
  "tweet_length", "word_count"
)


# Create modeling dataset
modeling_data <- train_grouped %>%
  select(tweet, final_label, all_of(features)) %>%
  rename(sexist = final_label) %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.) | is.nan(.) | is.infinite(.), 0, .)))

# Same features for dev data
dev_modeling <- dev_data %>%
  select(tweet, sexist, all_of(features)) %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.) | is.nan(.) | is.infinite(.), 0, .)))

# Handle Class Imbalance with SMOTE

# Prepare data for SMOTE
smote_data <- modeling_data %>% select(-tweet)

# Apply SMOTE to balance classes
smote_recipe <- recipe(sexist ~ ., data = smote_data) %>%
  step_smote(sexist)
smote_prep <- prep(smote_recipe)
train_data_balanced <- bake(smote_prep, new_data = NULL)

# Check balanced distribution
balanced_distribution <- train_data_balanced %>%
  count(sexist) %>%
  mutate(prop = n / sum(n))

print("Balanced Class Distribution after SMOTE:")
print(balanced_distribution)

# Visualize the balanced distribution
ggplot(balanced_distribution, aes(x = sexist, fill = sexist)) +
  geom_bar() +
  labs(title = "Class Distribution after SMOTE", 
       x = "Sexist Label", 
       y = "Count") +
  theme_minimal()


# Prepare final training and dev datasets

# Prepare features and target for training
X_train_final <- train_data_balanced %>% select(-sexist)
y_train_final <- train_data_balanced$sexist

# Prepare features and target for dev set
X_dev <- dev_modeling %>% 
  select(all_of(features)) %>%
  mutate(across(everything(), ~ ifelse(is.na(.), 0, .))) # Replace NA values with 0
y_dev <- as.factor(dev_modeling$sexist)

# Convert to data frames
X_train_df <- as.data.frame(X_train_final)
# Ensure consistent factor levels across all datasets
y_train_final <- factor(y_train_final, levels = c("no", "yes"))  

X_dev_df <- as.data.frame(X_dev)
y_dev <- factor(y_dev, levels = c("no", "yes"))

# Scale features - using training set parameters for dev set to prevent data leakage
X_train_scaled <- scale(X_train_df)
X_dev_scaled <- scale(X_dev_df, 
                      center = attr(X_train_scaled, "scaled:center"), 
                      scale = attr(X_train_scaled, "scaled:scale"))

# Convert to dataframes for model training
X_train_scaled_df <- as.data.frame(X_train_scaled)
X_dev_scaled_df <- as.data.frame(X_dev_scaled)

# Combine predictors and response for training
train_data_combined <- cbind(X_train_scaled_df, sexist = y_train_final)

# Set up for parallel processing to speed up cross-validation
num_cores <- detectCores() - 1  # Leave one core free
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# -------------------------------------------------------------------------------------------------------------------
# Model Training and Evaluation
# -------------------------------------------------------------------------------------------------------------------

# Cross-Validation Setup


# Define cross-validation control
cv_control <- trainControl(
  method = "cv",
  number = 5,  # 5-fold cross-validation
  classProbs = TRUE,
  summaryFunction = twoClassSummary,  # For binary classification metrics
  savePredictions = "final",
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Create list to store all ROC objects and prediction results
roc_list <- list()
prediction_list <- list()
cv_results_list <- list()

# Function to evaluate model performance on the dev set
evaluate_model <- function(model, X_dev, y_dev, model_name, prob_method = "prob") {
  if (prob_method == "prob") {
    # For models that use predict(..., type="prob")
    pred_probs <- predict(model, X_dev, type = "prob")
    pred_prob <- pred_probs[, "yes"]
  } else if (prob_method == "response") {
    # For models like glmnet that use predict(..., type="response")
    pred_prob <- predict(model, X_dev, type = "response")
  } else if (prob_method == "xgb") {
    # For XGBoost
    X_dev_matrix <- as.matrix(X_dev)
    pred_prob <- predict(model, newdata = X_dev_matrix)
  }
  
  # Calculate binary predictions using 0.5 threshold
  pred_class <- ifelse(pred_prob > 0.5, "yes", "no") %>% factor(levels = c("no", "yes"))
  
  # Calculate performance metrics
  conf_mat <- confusionMatrix(pred_class, y_dev)
  roc_obj <- roc(as.numeric(y_dev == "yes"), pred_prob)
  auc_val <- auc(roc_obj)
  
  # Store results
  roc_list[[model_name]] <<- roc_obj
  prediction_list[[model_name]] <<- list(
    pred_prob = pred_prob,
    pred_class = pred_class,
    conf_mat = conf_mat,
    auc = auc_val
  )
  
  # Print results
  cat("\n", model_name, "Results:\n")
  print(conf_mat)
  cat("AUC:", auc_val, "\n")
  
  return(list(conf_mat = conf_mat, auc = auc_val))
}

# -----------------------------------------------------------------------------
# MODEL 1: Logistic Regression with CV Tuning
# -----------------------------------------------------------------------------
cat("\nTraining Logistic Regression model with cross-validation...\n")

# Set up grid for alpha (elastic net mixing parameter) and lambda (regularization strength)
glmnet_grid <- expand.grid(
  alpha = c(0, 0.2, 0.5, 0.8, 1),  # 0 = ridge, 1 = lasso, between = elastic net
  lambda = 10^seq(-4, 0, length.out = 10)
)

# Train model with caret for cross-validation
lr_cv_model <- train(
  x = as.matrix(X_train_scaled_df),
  y = y_train_final,
  method = "glmnet",
  trControl = cv_control,
  tuneGrid = glmnet_grid,
  metric = "ROC"
)

# Extract best model parameters
lr_best_alpha <- lr_cv_model$bestTune$alpha
lr_best_lambda <- lr_cv_model$bestTune$lambda


cat("\nLogistic Regression Best Parameters:\n")
cat("Alpha:", lr_best_alpha, "\n")
cat("Lambda:", lr_best_lambda, "\n")

# Evaluate on dev set
lr_results <- evaluate_model(
  lr_cv_model, 
  X_dev_scaled_df, 
  y_dev, 
  "Logistic Regression", 
  prob_method = "prob"
)

conf_mat_lr <- lr_results$conf_mat
auc_lr <- lr_results$auc

# Store cross-validation results
cv_results_list[["Logistic Regression"]] <- lr_cv_model$results

# -----------------------------------------------------------------------------
# MODEL 2: Support Vector Machine with CV Tuning
# -----------------------------------------------------------------------------
cat("\nTraining SVM model with cross-validation...\n")

# Set up grid for SVM parameters
svm_grid <- expand.grid(
  sigma = 2^seq(-8, 1, length.out = 5),
  C = 2^seq(-2, 7, length.out = 5)
)

# Train model with caret for cross-validation
svm_cv_model <- train(
  x = X_train_scaled_df,
  y = y_train_final,
  method = "svmRadial",
  trControl = cv_control,
  tuneGrid = svm_grid,
  metric = "ROC",
  preProcess = NULL  # Already scaled
)

# Extract best model parameters
svm_best_sigma <- svm_cv_model$bestTune$sigma
svm_best_C <- svm_cv_model$bestTune$C


cat("\nSVM Best Parameters:\n")
cat("Sigma:", svm_best_sigma, "\n")
cat("C:", svm_best_C, "\n")

# Evaluate on dev set
svm_results <- evaluate_model(
  svm_cv_model, 
  X_dev_scaled_df, 
  y_dev, 
  "SVM", 
  prob_method = "prob"
)

conf_mat_svm <- svm_results$conf_mat
auc_svm <- svm_results$auc

# Store cross-validation results
cv_results_list[["SVM"]] <- svm_cv_model$results

# -----------------------------------------------------------------------------
# MODEL 3: Decision Tree with CV Tuning
# -----------------------------------------------------------------------------
cat("\nTraining Decision Tree model with cross-validation...\n")

# Set up grid for decision tree parameters
rpart_grid <- expand.grid(
  cp = seq(0.001, 0.05, by = 0.005)  # complexity parameter
)

# Train model with caret for cross-validation
dt_cv_model <- train(
  x = X_train_scaled_df,
  y = y_train_final,
  method = "rpart",
  trControl = cv_control,
  tuneGrid = rpart_grid,
  metric = "ROC"
)

# Extract best model parameters
dt_best_cp <- dt_cv_model$bestTune$cp

cat("\nDecision Tree Best Parameters:\n")
cat("Complexity Parameter:", dt_best_cp, "\n")

# Evaluate on dev set
dt_results <- evaluate_model(
  dt_cv_model, 
  X_dev_scaled_df, 
  y_dev, 
  "Decision Tree", 
  prob_method = "prob"
)

conf_mat_dt <- dt_results$conf_mat
auc_dt <- dt_results$auc

# Store cross-validation results
cv_results_list[["Decision Tree"]] <- dt_cv_model$results

# Plot the decision tree
best_tree <- dt_cv_model$finalModel
rpart.plot(best_tree, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)


# -----------------------------------------------------------------------------
# MODEL 4: Random Forest with CV Tuning
# -----------------------------------------------------------------------------
cat("\nTraining Random Forest model with cross-validation...\n")

# Set up grid for random forest parameters
rf_grid <- expand.grid(
  mtry = seq(2, min(20, ncol(X_train_scaled_df)), by = 3)
)

# Train model with caret for cross-validation
rf_cv_model <- train(
  x = X_train_scaled_df,
  y = y_train_final,
  method = "rf",
  trControl = cv_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  importance = TRUE,
  ntree = 300
)

# Extract best model parameters
rf_best_mtry <- rf_cv_model$bestTune$mtry

cat("\nRandom Forest Best Parameters:\n")
cat("mtry:", rf_best_mtry, "\n")

# Evaluate on dev set
rf_results <- evaluate_model(
  rf_cv_model, 
  X_dev_scaled_df, 
  y_dev, 
  "Random Forest", 
  prob_method = "prob"
)

conf_mat_rf <- rf_results$conf_mat
auc_rf <- rf_results$auc

# Store cross-validation results
cv_results_list[["Random Forest"]] <- rf_cv_model$results

# -----------------------------------------------------------------------------
# MODEL 5: XGBoost with CV Tuning
# -----------------------------------------------------------------------------
cat("\nTraining XGBoost model with cross-validation...\n")

# Set up xgb parameters grid
xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 6, 9),
  gamma = c(0, 0.1, 0.5),
  colsample_bytree = c(0.6, 0.8, 1.0),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.6, 0.8, 1.0)
)

# For simplicity in this example, we'll use a smaller grid
xgb_grid_small <- expand.grid(
  nrounds = c(100),
  eta = c(0.1),
  max_depth = c(6),
  gamma = c(0),
  colsample_bytree = c(0.8),
  min_child_weight = c(1),
  subsample = c(0.8)
)

# Create proper train/test matrices for XGBoost
X_train_matrix <- as.matrix(X_train_scaled_df)
X_dev_matrix <- as.matrix(X_dev_scaled_df)

# Train XGBoost model with cv
xgb_cv_model <- train(
  x = X_train_matrix,
  y = y_train_final,
  method = "xgbTree",
  trControl = cv_control,
  tuneGrid = xgb_grid_small,
  metric = "ROC",
  verbose = FALSE
)

# Extrair os melhores parâmetros do modelo
xgb_best_params <- xgb_cv_model$bestTune

cat("\nXGBoost Best Parameters:\n")
print(xgb_best_params)

# Predições no conjunto de validação (dev set)
probs_xgb <- predict(xgb_cv_model, newdata = X_dev_matrix, type = "prob")[, 2]
preds_xgb <- ifelse(probs_xgb > 0.5, levels(y_dev)[2], levels(y_dev)[1])
preds_xgb <- factor(preds_xgb, levels = levels(y_dev))

# Matriz de confusão
conf_mat_xgb <- caret::confusionMatrix(preds_xgb, y_dev, positive = levels(y_dev)[2])

# Calculando a curva ROC e o AUC utilizando o pacote pROC
roc_xgb <- pROC::roc(response = y_dev, predictor = probs_xgb)
auc_xgb <- pROC::auc(roc_xgb)

# Exibir o AUC no console
cat("\nXGBoost AUC:\n")
print(auc_xgb)

# Armazenar os resultados de cross-validation
cv_results_list[["XGBoost"]] <- xgb_cv_model$results

# -----------------------------------------------------------------------------
# MODEL 6: Gradient Boosting Machine with CV Tuning
# -----------------------------------------------------------------------------
cat("\nTraining GBM model with cross-validation...\n")

# Set up grid for GBM parameters
gbm_grid <- expand.grid(
  n.trees = c(100, 200),
  interaction.depth = c(3, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = c(10)
)

# Train model with caret for cross-validation
gbm_cv_model <- train(
  x = X_train_scaled_df,
  y = y_train_final,
  method = "gbm",
  trControl = cv_control,
  tuneGrid = gbm_grid,
  metric = "ROC",
  verbose = FALSE
)

# Extract best model parameters
gbm_best_params <- gbm_cv_model$bestTune

cat("\nGBM Best Parameters:\n")
print(gbm_best_params)

# Predictions on dev set
probs_gbm <- predict(gbm_cv_model, newdata = X_dev_scaled_df, type = "prob")[, 2]
preds_gbm <- ifelse(probs_gbm > 0.5, levels(y_dev)[2], levels(y_dev)[1])
preds_gbm <- factor(preds_gbm, levels = levels(y_dev))

# Confusion matrix
conf_mat_gbm <- caret::confusionMatrix(preds_gbm, y_dev, positive = levels(y_dev)[2])

# ROC and AUC
roc_gbm <- pROC::roc(response = y_dev, predictor = probs_gbm)
auc_gbm <- pROC::auc(roc_gbm)

# Output AUC to console
cat("\nGBM AUC:\n")
print(auc_gbm)

# Store cross-validation results
cv_results_list[["GBM"]] <- gbm_cv_model$results


# Model Comparison

# Create comprehensive model comparison dataframe
model_comparison <- data.frame(
  Model = c("Logistic Regression", "SVM", "Decision Tree", "Random Forest", 
            "XGBoost", "GBM"),
  Accuracy = c(conf_mat_lr$overall["Accuracy"], 
               conf_mat_svm$overall["Accuracy"], 
               conf_mat_dt$overall["Accuracy"],
               conf_mat_rf$overall["Accuracy"],
               conf_mat_xgb$overall["Accuracy"],
               conf_mat_gbm$overall["Accuracy"]),
  Precision = c(conf_mat_lr$byClass["Pos Pred Value"], 
                conf_mat_svm$byClass["Pos Pred Value"], 
                conf_mat_dt$byClass["Pos Pred Value"],
                conf_mat_rf$byClass["Pos Pred Value"],
                conf_mat_xgb$byClass["Pos Pred Value"],
                conf_mat_gbm$byClass["Pos Pred Value"]),
  Recall = c(conf_mat_lr$byClass["Sensitivity"], 
             conf_mat_svm$byClass["Sensitivity"], 
             conf_mat_dt$byClass["Sensitivity"],
             conf_mat_rf$byClass["Sensitivity"],
             conf_mat_xgb$byClass["Sensitivity"],
             conf_mat_gbm$byClass["Sensitivity"]),
  F1_Score = c(conf_mat_lr$byClass["F1"], 
               conf_mat_svm$byClass["F1"], 
               conf_mat_dt$byClass["F1"],
               conf_mat_rf$byClass["F1"],
               conf_mat_xgb$byClass["F1"],
               conf_mat_gbm$byClass["F1"]),
  AUC = c(auc_lr, auc_svm, auc_dt, auc_rf, auc_xgb, auc_gbm)
)

print("Final Model Comparison:")
print(model_comparison)

# VISUALIZATIONS

# ROC Curve Plot for all models

plot(roc_lr, col = "blue", main = "ROC Curves Comparison", lwd = 2)
plot(roc_svm, col = "red", add = TRUE, lwd = 2)
plot(roc_dt, col = "green", add = TRUE, lwd = 2)
plot(roc_rf, col = "orange", add = TRUE, lwd = 2)
plot(roc_xgb, col = "purple", add = TRUE, lwd = 2)
plot(roc_gbm, col = "brown", add = TRUE, lwd = 2)
legend("bottomright", 
       legend = c("Logistic Regression", "SVM", "Decision Tree", "Random Forest", 
                  "XGBoost", "GBM"),
       col = c("blue", "red", "green", "orange", "purple", "brown", "pink", "darkgray"), 
       lwd = 2)


# Model Comparison Bar Chart
model_comparison_long <- model_comparison %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score, AUC),
               names_to = "Metric", values_to = "Value")

ggplot(model_comparison_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Model Performance Comparison", x = "Model", y = "Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set1")


# Confusion Matrix Visualization for Best Model
# Find the best model based on AUC
best_model_name <- model_comparison$Model[which.max(model_comparison$AUC)]
best_conf_mat <- get(paste0("conf_mat_", tolower(gsub(" ", "", best_model_name))))

conf_mat_df <- as.data.frame(best_conf_mat$table)
colnames(conf_mat_df) <- c("Reference", "Prediction", "Freq")

ggplot(conf_mat_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 12) +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = paste("Confusion Matrix -", best_model_name), x = "Actual", y = "Predicted") +
  theme_minimal()

