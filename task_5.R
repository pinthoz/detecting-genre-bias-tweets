# Task 5 - Predicting Annotator Decisions (Recommendation System Approach)

# Load necessary libraries
library(recommenderlab)
library(dplyr)
library(Matrix)
library(Metrics)
library(caret)
library(ggplot2)
library(pROC) # For AUC and ROC curves
library(gridExtra) # For arranging ggplots

# --- 1. Enhanced Data Loading and Preprocessing ---
data <- read.csv("data/EXIST2025_train.csv", stringsAsFactors = FALSE)
data$label_numeric <- ifelse(data$label_task1_1 == "YES", 1, 0)

# Create comprehensive annotator and tweet mappings
annotator_mapping <- data %>%
  select(annotator_id) %>%
  distinct() %>%
  mutate(annotator_numeric = row_number())

tweet_mapping <- data %>%
  select(id_EXIST) %>%
  distinct() %>%
  mutate(tweet_numeric = row_number())

# Enhanced data with proper mappings
data_enhanced <- data %>%
  left_join(annotator_mapping, by = "annotator_id") %>%
  left_join(tweet_mapping, by = "id_EXIST")

# --- 2. Profile Calculation Function ---
# Function to calculate annotator and tweet profiles from a given dataset
calculate_profiles_from_data <- function(input_df) {
  # input_df should have columns: annotator_id, annotator_numeric, id_EXIST, tweet_numeric, label_numeric
  if (!all(c("annotator_id", "annotator_numeric", "id_EXIST", "tweet_numeric", "label_numeric") %in% names(input_df))) {
    stop("Input DataFrame for profile calculation is missing required columns.")
  }
  
  annotator_profiles_df <- input_df %>%
    group_by(annotator_id, annotator_numeric) %>%
    summarise(
      total_annotations = n(),
      sexist_rate = mean(label_numeric),
      consistency_var = if(n() > 1) var(label_numeric) else NA_real_, # var() needs >1 obs
      .groups = 'drop'
    )
  # Ensure consistency_var is NA or a value, not NaN if var somehow produced it with n=1 earlier
  annotator_profiles_df$consistency_var[is.nan(annotator_profiles_df$consistency_var)] <- NA_real_
  
  tweet_profiles_df <- input_df %>%
    group_by(id_EXIST, tweet_numeric) %>%
    summarise(
      annotation_count = n(),
      agreement_rate = mean(label_numeric), # Tweet's average "sexist" score
      disagreement = if(n() > 1) var(label_numeric) else NA_real_,    # Variance in labels
      difficulty = abs(0.5 - agreement_rate), # Scores near 0.5 are more ambiguous
      .groups = 'drop'
    )
  tweet_profiles_df$disagreement[is.nan(tweet_profiles_df$disagreement)] <- NA_real_
  
  return(list(annotator_profiles = annotator_profiles_df, tweet_profiles = tweet_profiles_df))
}

# Calculate global profiles for initial analysis (using all data)
global_profiles <- calculate_profiles_from_data(data_enhanced)
global_annotator_profiles <- global_profiles$annotator_profiles
global_tweet_profiles <- global_profiles$tweet_profiles

# Enhanced rating matrix with better handling
rating_matrix <- sparseMatrix(
  i = data_enhanced$annotator_numeric,
  j = data_enhanced$tweet_numeric,
  x = data_enhanced$label_numeric,
  dims = c(nrow(annotator_mapping), nrow(tweet_mapping)), # Use mapping counts for precise dims
  dimnames = list(paste0("u", annotator_mapping$annotator_numeric), 
                  paste0("t", tweet_mapping$tweet_numeric))
)
rrm <- new("realRatingMatrix", data = rating_matrix)

# --- 3. Enhanced Data Analysis (using global profiles) ---
cat("=== Dataset Analysis ===\n")
cat("Number of annotators:", nrow(global_annotator_profiles), "\n")
cat("Number of tweets:", nrow(global_tweet_profiles), "\n")
cat("Total annotations:", nrow(data_enhanced), "\n")
cat("Overall sexist rate:", round(mean(data_enhanced$label_numeric), 3), "\n")
cat("Matrix sparsity:", round(1 - length(getRatings(rrm)) / (nrow(rrm) * ncol(rrm)), 4) * 100, "%\n")

# Visualizations
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1)) # Adjusted margins
hist(global_annotator_profiles$sexist_rate, main = "Annotator Bias (Global)", xlab = "Prop. Sexist Labels", col = "lightblue", breaks = 20)
hist(global_tweet_profiles$agreement_rate, main = "Tweet Agreement (Global)", xlab = "Prop. Sexist Labels", col = "lightgreen", breaks = 20)
hist(global_annotator_profiles$total_annotations, main = "Annotations/Annotator (Global)", xlab = "# Annotations", col = "orange", breaks = 20)
hist(global_tweet_profiles$difficulty, main = "Tweet Difficulty (Global)", xlab = "Difficulty Score", col = "pink", breaks = 20)
par(mfrow = c(1, 1))

# --- 4. Comprehensive Train/Test Split Strategy ---
set.seed(42)
# Traditional train-test split for the original dataframe
train_indices <- createDataPartition(data_enhanced$label_numeric, p = 0.8, list = FALSE)
train_data <- data_enhanced[train_indices, ]
test_data <- data_enhanced[-train_indices, ]

cat("=== Train-Test Split Summary ===\n")
cat("Training set size:", nrow(train_data), "annotations\n")
cat("Test set size:", nrow(test_data), "annotations\n")

# Calculate profiles for feature generation STRICTLY from train_data
training_set_profiles <- calculate_profiles_from_data(train_data)
annotator_profiles_for_feature_gen <- training_set_profiles$annotator_profiles
tweet_profiles_for_feature_gen <- training_set_profiles$tweet_profiles

# Create rating matrices for train and test
train_rating_matrix <- sparseMatrix(
  i = train_data$annotator_numeric, j = train_data$tweet_numeric, x = train_data$label_numeric,
  dims = c(nrow(annotator_mapping), nrow(tweet_mapping)),
  dimnames = list(paste0("u", annotator_mapping$annotator_numeric), paste0("t", tweet_mapping$tweet_numeric))
)

train_rrm <- new("realRatingMatrix", data = train_rating_matrix)

# Recommenderlab evaluation scheme for collaborative filtering model validation
# This splits the train_rrm further.
eval_scheme <- evaluationScheme(train_rrm, method = "split", train = 0.75, given = -1, goodRating = 1)
cf_train_rrm <- getData(eval_scheme, "train")
cf_test_known <- getData(eval_scheme, "known")
cf_test_unknown <- getData(eval_scheme, "unknown")

# Calculate profiles for CF evaluation STRICTLY from cf_train_rrm data
cf_train_df_raw <- as(cf_train_rrm, "data.frame")
colnames(cf_train_df_raw) <- c("annotator_numeric_str", "tweet_numeric_str", "label_numeric")
cf_train_df <- cf_train_df_raw %>%
  mutate(
    annotator_numeric = as.numeric(gsub("u", "", annotator_numeric_str)),
    tweet_numeric = as.numeric(gsub("t", "", tweet_numeric_str))
  ) %>%
  left_join(annotator_mapping %>% select(annotator_id, annotator_numeric), by = "annotator_numeric") %>%
  left_join(tweet_mapping %>% select(id_EXIST, tweet_numeric), by = "tweet_numeric")

cf_evaluation_profiles <- calculate_profiles_from_data(cf_train_df)
annotator_profiles_for_cf_eval <- cf_evaluation_profiles$annotator_profiles
# tweet_profiles_for_cf_eval <- cf_evaluation_profiles$tweet_profiles # If needed


# --- 5. Model Training with Parameter Tuning ---
cat("\n=== Training Enhanced Models (on CF training set) ===\n")
# Note: Parameters presented here are assumed to be pre-tuned or reasonable defaults.
# Proper hyperparameter tuning would involve cross-validation on the cf_train_rrm.
model_popular <- Recommender(cf_train_rrm, method = "POPULAR")
model_ubcf <- Recommender(cf_train_rrm, method = "UBCF", parameter = list(method = "cosine", nn = 25, normalize = "center"))
model_ibcf <- Recommender(cf_train_rrm, method = "IBCF", parameter = list(method = "cosine", k = 30, normalize = "center"))
model_svd <- Recommender(cf_train_rrm, method = "SVD", parameter = list(k = 25, gamma = 0.015, lambda = 0.001))
model_random <- Recommender(cf_train_rrm, method = "RANDOM")
cat("Base models trained successfully.\n")

# --- 6. Advanced Prediction Generation ---
cat("\n=== Generating Predictions (for CF evaluation) ===\n")
pred_popular <- predict(model_popular, cf_test_known, type = "ratings")
pred_ubcf <- predict(model_ubcf, cf_test_known, type = "ratings")
pred_ibcf <- predict(model_ibcf, cf_test_known, type = "ratings")
pred_svd <- predict(model_svd, cf_test_known, type = "ratings")
pred_random <- predict(model_random, cf_test_known, type = "ratings")

# Enhanced dense matrix conversion
get_dense_matrix_safe <- function(pred_obj, ref_dimnames, default_value = 0) {
  full_m <- matrix(default_value, nrow = length(ref_dimnames[[1]]), ncol = length(ref_dimnames[[2]]),
                   dimnames = ref_dimnames)
  if (!is.null(pred_obj) && length(getRatings(pred_obj)) > 0) {
    pred_df <- as(pred_obj, "data.frame") # user, item, rating
    for (k in 1:nrow(pred_df)) {
      if (pred_df[k,1] %in% ref_dimnames[[1]] && pred_df[k,2] %in% ref_dimnames[[2]]) {
        full_m[pred_df[k,1], pred_df[k,2]] <- pred_df[k,3]
      }
    }
  }
  full_m[is.na(full_m)] <- default_value # Should not happen if default_value is numeric
  return(full_m)
}

target_dimnames <- dimnames(cf_test_known) # Use dimnames for robust alignment

mat_popular <- get_dense_matrix_safe(pred_popular, target_dimnames, 0.5) # Use 0.5 as neutral for missing preds
mat_ubcf    <- get_dense_matrix_safe(pred_ubcf,    target_dimnames, 0.5)
mat_ibcf    <- get_dense_matrix_safe(pred_ibcf,    target_dimnames, 0.5)
mat_svd     <- get_dense_matrix_safe(pred_svd,     target_dimnames, 0.5)
mat_random  <- get_dense_matrix_safe(pred_random,  target_dimnames, 0.5)

# --- 7. Advanced Ensemble Methods (for CF evaluation) ---
mat_hybrid_simple <- 0.35 * mat_popular + 0.25 * mat_ubcf + 0.25 * mat_ibcf + 0.15 * mat_svd

adjust_for_annotator_bias <- function(pred_matrix, current_annotator_profiles) {
  adjusted_matrix <- pred_matrix # inherits dimnames
  if (is.null(current_annotator_profiles) || nrow(current_annotator_profiles) == 0) {
    warning("Annotator profiles for bias adjustment are missing or empty. No bias adjustment will be applied.")
    return(pmax(0, pmin(1, adjusted_matrix))) # Return original (clipped) if no profiles
  }
  
  for (i in 1:nrow(pred_matrix)) { # i is row index
    annotator_numeric_val <- as.numeric(gsub("u", "", rownames(pred_matrix)[i]))
    
    profile_info <- current_annotator_profiles %>% 
      filter(annotator_numeric == annotator_numeric_val)
    
    if (nrow(profile_info) > 0) {
      bias <- profile_info$sexist_rate[1] # Get the bias
      # Only adjust if bias is a valid, non-NA number
      if(!is.na(bias) && is.finite(bias)){ 
        adjusted_matrix[i, ] <- 0.7 * pred_matrix[i, ] + 0.3 * bias
      } # Else: if bias is NA or not finite, keep original pred_matrix[i, ] for this annotator
    } # Else: if no profile_info for this annotator_numeric_val, keep original
  }
  return(pmax(0, pmin(1, adjusted_matrix))) # Ensure [0,1] range
}

# Use annotator_profiles_for_cf_eval for this evaluation step
mat_hybrid_bias_adjusted <- adjust_for_annotator_bias(mat_hybrid_simple, annotator_profiles_for_cf_eval)

# Coerce to matrix (if not already), with correct dimnames
mat_hybrid_bias_adjusted <- matrix(mat_hybrid_bias_adjusted, 
                                   nrow = nrow(mat_hybrid_simple), 
                                   ncol = ncol(mat_hybrid_simple), 
                                   dimnames = dimnames(cf_test_known))

calculate_prediction_confidence <- function(matrices_list) {
  pred_array <- array(unlist(matrices_list), dim = c(nrow(matrices_list[[1]]), ncol(matrices_list[[1]]), length(matrices_list)))
  pred_var <- apply(pred_array, c(1, 2), var, na.rm = TRUE)
  pred_var[is.na(pred_var)] <- 0 # If only one model or all same, var is NA, treat as 0 variance
  confidence <- 1 / (1 + pred_var) 
  return(confidence)
}
all_pred_matrices_for_confidence <- list(mat_popular, mat_ubcf, mat_ibcf, mat_svd)
confidence_map <- calculate_prediction_confidence(all_pred_matrices_for_confidence)

# Weighted average as base for confidence ensemble
ensemble_avg <- Reduce("+", all_pred_matrices_for_confidence) / length(all_pred_matrices_for_confidence)
# Heuristic for confidence: more confident predictions get higher weight from ensemble_avg
# Less confident predictions might rely more on a stable baseline like POPULAR or simple average.
# Original heuristic: mat_hybrid_confidence <- ensemble_avg * confidence_map + mat_popular * (1 - confidence_map) * 0.5
# Alternative: Use ensemble_avg for confident, simple_hybrid for less confident.
mat_hybrid_confidence <- ensemble_avg * confidence_map + mat_hybrid_simple * (1 - confidence_map)
mat_hybrid_confidence <- pmax(0, pmin(1, mat_hybrid_confidence))
mat_hybrid_confidence <- matrix(mat_hybrid_confidence,
                                nrow = nrow(mat_hybrid_simple),
                                ncol = ncol(mat_hybrid_simple),
                                dimnames = dimnames(cf_test_known))


# --- 8. Enhanced Evaluation Functions ---
get_aligned_ratings_enhanced <- function(pred_input, true_rrm_unknown) {
  # ... (previous version of this function was largely okay, ensuring dimnames align)
  # The key is that true_df comes from true_rrm_unknown which has user/item dimnames
  # and pred_input (if matrix) also has these dimnames.
  true_df <- as(true_rrm_unknown, "data.frame") # user, item, rating
  colnames(true_df) <- c("user", "item", "rating_true")
  
  aligned_preds <- numeric(0)
  aligned_trues <- numeric(0)
  
  if (is(pred_input, "realRatingMatrix")) {
    if (length(getRatings(pred_input)) > 0) {
      pred_df <- as(pred_input, "data.frame")
      colnames(pred_df) <- c("user", "item", "rating_pred")
      eval_df <- merge(pred_df, true_df, by = c("user", "item"))
      if(nrow(eval_df) > 0) {
        aligned_preds <- eval_df$rating_pred
        aligned_trues <- eval_df$rating_true
      }
    }
  } else if (is.matrix(pred_input)) {
    # Assumes pred_input matrix has dimnames matching true_df$user and true_df$item
    temp_preds <- numeric(nrow(true_df))
    valid_count <- 0
    for (k in 1:nrow(true_df)) {
      user_val <- true_df$user[k]
      item_val <- true_df$item[k]
      if (user_val %in% rownames(pred_input) && item_val %in% colnames(pred_input)) {
        temp_preds[k] <- pred_input[user_val, item_val]
        valid_count <- valid_count + 1
      } else {
        temp_preds[k] <- NA # Mark as NA if not found
      }
    }
    if (valid_count > 0) {
      aligned_preds <- temp_preds[!is.na(temp_preds)]
      aligned_trues <- true_df$rating_true[!is.na(temp_preds)]
    }
  }
  
  if (length(aligned_preds) == 0) {
    warning("No common ratings found for alignment in get_aligned_ratings_enhanced.")
  }
  return(list(preds = aligned_preds, trues = aligned_trues))
}


evaluate_model_comprehensive <- function(aligned_ratings, model_name) {
  if (length(aligned_ratings$preds) == 0 || length(aligned_ratings$trues) == 0 || 
      length(aligned_ratings$preds) != length(aligned_ratings$trues)) {
    warning(paste("Insufficient or mismatched aligned ratings for model:", model_name))
    return(data.frame(Model = model_name, Threshold = NA, RMSE = NA, Accuracy = NA, Precision = NA, Recall = NA, F1 = NA, AUC = NA, stringsAsFactors = FALSE))
  }
  
  rmse_val <- rmse(aligned_ratings$trues, aligned_ratings$preds)
  auc_val <- tryCatch({
    if (length(unique(aligned_ratings$trues)) < 2) {NA_real_} 
    else { 
      roc_obj <- roc(response = aligned_ratings$trues, 
                     predictor = aligned_ratings$preds, 
                     quiet = TRUE, 
                     levels = c(0,1), # Control group is 0, Case group is 1
                     direction = "<") # CORRECTED: Predictor is higher for cases (1) than controls (0)
      as.numeric(auc(roc_obj))
    }
  }, error = function(e) {
    warning(paste("AUC calculation failed for model:", model_name, "Error:", e$message))
    NA_real_ 
  })
  
  thresholds <- c(0.3, 0.4, 0.5, 0.6, 0.7)
  results_list <- lapply(thresholds, function(thresh) {
    pred_bin <- factor(ifelse(aligned_ratings$preds > thresh, 1, 0), levels = c(0, 1))
    true_bin <- factor(aligned_ratings$trues, levels = c(0, 1)) # Assuming trues are already 0/1
    
    cm_obj <- try(confusionMatrix(pred_bin, true_bin, positive = "1"), silent = TRUE)
    if(inherits(cm_obj, "try-error")){
      return(data.frame(Model=model_name, Threshold=thresh, RMSE=rmse_val, Accuracy=NA, Precision=NA, Recall=NA, F1=NA, AUC=auc_val))
    }
    
    data.frame(
      Model = model_name, Threshold = thresh, RMSE = rmse_val,
      Accuracy = cm_obj$overall["Accuracy"],
      Precision = ifelse(is.nan(cm_obj$byClass["Precision"]), 0, cm_obj$byClass["Precision"]), # Handle NaN by Precision = TP / (TP+FP)
      Recall = cm_obj$byClass["Recall"], F1 = cm_obj$byClass["F1"], AUC = auc_val )
  })
  return(do.call(rbind, results_list))
}

# --- 9. Comprehensive Model Evaluation (on CF test set) ---
cat("\n=== Comprehensive Model Evaluation (CF Split) ===\n")
models_to_evaluate <- list(
  "POPULAR" = pred_popular, "UBCF" = pred_ubcf, "IBCF" = pred_ibcf, "SVD" = pred_svd, "RANDOM" = pred_random,
  "HYBRID_SIMPLE" = mat_hybrid_simple,
  "HYBRID_BIAS_ADJ" = mat_hybrid_bias_adjusted, # Renamed for plot clarity
  "HYBRID_CONF" = mat_hybrid_confidence     # Renamed
)
all_results <- data.frame()
for (model_name in names(models_to_evaluate)) {
  cat("Evaluating (CF):", model_name, "\n")
  aligned_data <- get_aligned_ratings_enhanced(models_to_evaluate[[model_name]], cf_test_unknown)
  model_results <- evaluate_model_comprehensive(aligned_data, model_name)
  all_results <- rbind(all_results, model_results)
}


# --- 10. Results Analysis and Visualization (CF Evaluation) ---
cat("\n=== Performance Summary (CF Evaluation) ===\n")
best_by_threshold <- all_results %>% filter(!is.na(F1)) %>% group_by(Threshold) %>% slice_max(F1, n = 1) %>% ungroup()
print("Best F1 Score by Threshold (CF Eval):"); print(best_by_threshold)
best_overall_cf <- all_results %>% filter(Threshold == 0.5, !is.na(F1)) %>% arrange(desc(F1)) %>% head(1)
cat("\nBest Overall Model (CF Eval at Threshold 0.5):\n"); print(best_overall_cf)

# Visualizations
p1 <- ggplot(all_results %>% filter(!is.na(F1)), aes(x = Threshold, y = F1, color = Model, group = Model)) +
  geom_line(size = 1) + geom_point(size = 2) + labs(title = "F1 Score vs Threshold (CF Eval)", y = "F1 Score") + theme_minimal() + theme(legend.position = "bottom")

auc_summary <- all_results %>% distinct(Model, .keep_all = TRUE) %>% select(Model, AUC) %>% filter(!is.na(AUC))
p2 <- ggplot(auc_summary, aes(x = reorder(Model, AUC), y = AUC)) + geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + labs(title = "AUC Comparison (CF Eval)", x = "Model") + theme_minimal()

rmse_summary <- all_results %>% distinct(Model, .keep_all = TRUE) %>% select(Model, RMSE) %>% filter(!is.na(RMSE))
p3 <- ggplot(rmse_summary, aes(x = reorder(Model, -RMSE), y = RMSE)) + geom_bar(stat = "identity", fill = "coral") +
  coord_flip() + labs(title = "RMSE Comparison (CF Eval)", x = "Model") + theme_minimal()

grid.arrange(p1, p2, p3, ncol = 2)


# --- 11. Feature Integration for Main Classifier ---
cat("\n=== Generating Features for Main Classifier (using full train_data) ===\n")
cat("Training full models on train_rrm for feature generation...\n")
# Models are trained on train_rrm (derived from 80% train_data)
full_model_popular <- Recommender(train_rrm, method = "POPULAR") # train_rrm is the full training data matrix
full_model_ubcf    <- Recommender(train_rrm, method = "UBCF", parameter = list(method = "cosine", nn = 25, normalize = "center"))
full_model_ibcf    <- Recommender(train_rrm, method = "IBCF", parameter = list(method = "cosine", k = 30, normalize = "center"))
full_model_svd     <- Recommender(train_rrm, method = "SVD",  parameter = list(k = 25, gamma = 0.015, lambda = 0.001))

generate_recommendation_features <- function(target_data_df, # e.g. train_data, test_data, dev_data
                                             trained_models_list, 
                                             annot_profiles_train, # Annotator profiles from training set
                                             tweet_profiles_train, # Tweet profiles from training set
                                             global_annot_map, global_tweet_map,
                                             data_type_label = "test") {
  # Adiciona identificador de linha
  target_data_df <- target_data_df %>% mutate(row_id = row_number())
  cat("Generating", data_type_label, "features for", nrow(target_data_df), "annotations...\n")
  
  # Cria matriz de predição "vazia" com todos os usuários e tweets conhecidos
  target_matrix_shell <- sparseMatrix(
    i = target_data_df$annotator_numeric,
    j = target_data_df$tweet_numeric,
    x = rep(NA_real_, nrow(target_data_df)),  
    dims = c(nrow(global_annot_map), nrow(global_tweet_map)),
    dimnames = list(
      paste0("u", global_annot_map$annotator_numeric), 
      paste0("t", global_tweet_map$tweet_numeric)
    )
  )
  target_rrm_shell <- new("realRatingMatrix", data = target_matrix_shell)
  target_rrm_shell <- new("realRatingMatrix", data = target_matrix_shell)
  
  target_dimnames_feat <- dimnames(target_rrm_shell)
  pred_features_df <- data.frame(row_id = 1:nrow(target_data_df))
  
  # Geração de features por modelo
  for (model_name in names(trained_models_list)) {
    cat("  Predicting with", model_name, "for", data_type_label, "set...\n")
    preds_obj <- predict(trained_models_list[[model_name]], target_rrm_shell, type = "ratings")
    pred_dense_m <- get_dense_matrix_safe(preds_obj, target_dimnames_feat, 0.5)
    
    feature_values <- sapply(1:nrow(target_data_df), function(k) {
      ann_idx_str <- paste0("u", target_data_df$annotator_numeric[k])
      tweet_idx_str <- paste0("t", target_data_df$tweet_numeric[k])
      if (ann_idx_str %in% rownames(pred_dense_m) && tweet_idx_str %in% colnames(pred_dense_m)) {
        return(pred_dense_m[ann_idx_str, tweet_idx_str])
      } else {
        return(0.5) # valor neutro
      }
    })
    
    pred_features_df[[paste0(model_name, "_pred")]] <- feature_values
  }
  
  # Adiciona viés do anotador
  pred_features_df <- pred_features_df %>%
    left_join(target_data_df %>% select(row_id, annotator_numeric), by = "row_id") %>%
    left_join(annot_profiles_train %>% select(annotator_numeric, annotator_bias = sexist_rate), by = "annotator_numeric") %>%
    mutate(annotator_bias = ifelse(is.na(annotator_bias), 0.5, annotator_bias)) # padrão neutro
  
  # Adiciona dificuldade e concordância do tweet
  pred_features_df <- pred_features_df %>%
    left_join(target_data_df %>% select(row_id, tweet_numeric), by = "row_id") %>%
    left_join(tweet_profiles_train %>% select(tweet_numeric, tweet_difficulty = difficulty, tweet_agreement = agreement_rate), by = "tweet_numeric") %>%
    mutate(
      tweet_difficulty = ifelse(is.na(tweet_difficulty), 0.5, tweet_difficulty),
      tweet_agreement = ifelse(is.na(tweet_agreement), 0.5, tweet_agreement)
    )
  
  # Limpa colunas temporárias
  pred_features_df <- pred_features_df %>%
    select(-row_id, -annotator_numeric, -tweet_numeric)
  
  return(pred_features_df)
}


# Generate features for the main test set (test_data)
models_for_features <- list(
  "POPULAR_Full" = full_model_popular, "UBCF_Full" = full_model_ubcf,
  "IBCF_Full" = full_model_ibcf, "SVD_Full" = full_model_svd
)

# Use annotator_profiles_for_feature_gen and tweet_profiles_for_feature_gen calculated from train_data
test_rec_features <- generate_recommendation_features(
  test_data, models_for_features,
  annotator_profiles_for_feature_gen, tweet_profiles_for_feature_gen,
  annotator_mapping, tweet_mapping, # Pass global mappings for dims
  "test"
)

# Also generate features for the main training set (train_data) for consistent feature sets
# Important: Models are already trained on train_rrm, so this is like getting 'in-sample' scores
# For some models (like kNN based), this might result in perfect or near-perfect recall for already rated items.
# It's often better to use k-fold cross-validation to generate out-of-sample predictions for the training set.
# However, for simplicity here, we get direct predictions. Be mindful of potential overfitting if using these directly.
cat("\nNote: Generating 'in-sample' recommendation features for the training set.\n")
cat("For more robust training features, consider k-fold cross-validated predictions.\n")
train_rec_features <- generate_recommendation_features(
  train_data, models_for_features,
  annotator_profiles_for_feature_gen, tweet_profiles_for_feature_gen,
  annotator_mapping, tweet_mapping,
  "train"
)

# Create ensemble features for both train and test feature sets
create_ensemble_features <- function(df, annot_bias_col_name = "annotator_bias") {
  # Ensure columns exist before trying to use them
  required_preds <- c("POPULAR_Full_pred", "UBCF_Full_pred", "IBCF_Full_pred", "SVD_Full_pred")
  if (!all(required_preds %in% names(df))) {
    stop("Missing one or more required base model prediction columns for ensemble features.")
  }
  if (!annot_bias_col_name %in% names(df)) {
    warning(paste("Annotator bias column", annot_bias_col_name, "not found. Bias adjustment will not be effective."))
    df[[annot_bias_col_name]] <- 0.5 # Add a default if missing
  }
  
  df$HYBRID_SIMPLE_feat <- 0.35 * df$POPULAR_Full_pred + 0.25 * df$UBCF_Full_pred + 
    0.25 * df$IBCF_Full_pred + 0.15 * df$SVD_Full_pred
  
  df$HYBRID_BIAS_ADJ_feat <- 0.7 * df$HYBRID_SIMPLE_feat + 0.3 * df[[annot_bias_col_name]]
  df$HYBRID_BIAS_ADJ_feat <- pmax(0, pmin(1, df$HYBRID_BIAS_ADJ_feat)) # Clip to [0,1]
  return(df)
}

train_rec_features <- create_ensemble_features(train_rec_features)
test_rec_features  <- create_ensemble_features(test_rec_features)

# Combine with original train/test data
train_data_with_features <- cbind(train_data, train_rec_features)
test_data_with_features  <- cbind(test_data,  test_rec_features)

remove_constant_prediction_columns <- function(df) {
  pred_cols <- grep("_pred$", names(df), value = TRUE)
  for (col in pred_cols) {
    if (all(df[[col]] == 0.5)) {
      message(paste("Removing constant column:", col))
      df[[col]] <- NULL
    }
  }
  return(df)
}

train_rec_features <- remove_constant_prediction_columns(train_rec_features)
test_rec_features  <- remove_constant_prediction_columns(test_rec_features)

head(train_rec_features)


# For dev dataset:

dev_data <- read.csv("data/EXIST2025_dev_labeled.csv", stringsAsFactors = FALSE)

dev_data$label_numeric <- ifelse(dev_data$label_task1_1 == "YES", 1, 0)

dev_data <- dev_data %>%
  left_join(annotator_mapping, by = "annotator_id") %>%
  left_join(tweet_mapping, by = "id_EXIST")

dev_rec_features <- generate_recommendation_features(
  dev_data, models_for_features,
  annotator_profiles_for_feature_gen, tweet_profiles_for_feature_gen,
  annotator_mapping, tweet_mapping,
  "dev"
)

dev_rec_features <- create_ensemble_features(dev_rec_features)

dev_data_with_features <- cbind(dev_data, dev_rec_features)

cat("Sample of DEV features (first few rows, first few columns):\n")
print(head(dev_data_with_features[, (ncol(dev_data)+1):min(ncol(dev_data_with_features), ncol(dev_data)+5)], 3))
