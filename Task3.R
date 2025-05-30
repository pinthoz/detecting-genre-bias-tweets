# --------------------------------------------------
# TASK 3: Annotator Clustering with Evaluation
# --------------------------------------------------

library(tidyverse)
library(cluster)
library(factoextra)
library(fpc)
library(fastDummies)
library(data.table)
library(countrycode)
library(clusterSim)
library(clusterCrit)
library(entropy) 
library(d)
# Load Dataset
df <- read.csv("data/EXIST2025_train.csv")

# Compute Annotator Behavior Features
annotator_stats <- df %>%
  group_by(annotator_id) %>%
  summarise(
    n_labeled = n(),
    n_sexist = sum(label_task1_1 == "YES"),
    n_non_sexist = sum(label_task1_1 == "NO"),
    sexist_rate = n_sexist / n_labeled
  )

# Compute Agreement with Majority Labels
majority_labels <- df %>%
  group_by(id_EXIST) %>%
  summarise(
    majority_label = case_when(
      sum(label_task1_1 == "YES") > sum(label_task1_1 == "NO") ~ "YES",
      sum(label_task1_1 == "NO") > sum(label_task1_1 == "YES") ~ "NO",
      TRUE ~ NA_character_
    )
  ) %>% filter(!is.na(majority_label))

agreement <- df %>%
  inner_join(majority_labels, by = "id_EXIST") %>%
  mutate(agrees = label_task1_1 == majority_label) %>%
  group_by(annotator_id) %>%
  summarise(agreement_rate = mean(agrees))

annotator_stats <- annotator_stats %>%
  left_join(agreement, by = "annotator_id")

# Add Demographics and Region (keep all annotations)
df$region <- countrycode(df$country, origin = 'country.name', destination = 'region')
df$region[is.na(df$region)] <- "Other"

# One-Hot Encode Demographics for all annotations
df_encoded <- fastDummies::dummy_cols(df,
                                      select_columns = c("gender", "age", "education", "ethnicity", "region"),
                                      remove_first_dummy = TRUE,
                                      remove_selected_columns = TRUE)

# Merge with annotator stats
df_encoded <- df_encoded %>%
  left_join(annotator_stats, by = "annotator_id")

# Select Features
features <- df_encoded %>%
  dplyr::select(n_labeled, sexist_rate, agreement_rate, 
                starts_with("gender_"), starts_with("age_"),
                starts_with("education_"), starts_with("ethnicity_"), 
                starts_with("region_")) %>%
  drop_na()
# Scale
features_scaled <- scale(features)

# Elbow method
fviz_nbclust(features_scaled, kmeans, method = "wss") + 
  ggtitle("Elbow Method for Optimal K (K-Means)") # 4

# Silhouette method
fviz_nbclust(features_scaled, kmeans, method = "silhouette") + 
  ggtitle("Silhouette Method for Optimal K (K-Means)") # 4

# K-Means Clustering
set.seed(123)
kmeans_res <- kmeans(features_scaled, centers = 4, nstart = 25)
df_encoded$kmeans_cluster <- as.factor(kmeans_res$cluster)

# Cluster Visualization
fviz_cluster(kmeans_res, data = features_scaled, main = "K-Means Clusters")

# K-Means Cluster Evaluation
kmeans_stats <- cluster.stats(dist(features_scaled), kmeans_res$cluster)

cat("K-means Cluster Evaluation:\n")
cat("Within cluster sum of squares:", kmeans_stats$within.cluster.ss, "\n")
cat("Average silhouette width:", kmeans_stats$avg.silwidth, "\n")
cat("Calinski-Harabasz index:", kmeans_stats$ch, "\n")
cat("Dunn index:", kmeans_stats$dunn, "\n")

# Hierarchical Clustering
dist_matrix <- dist(features_scaled)
hc_res <- hclust(dist_matrix, method = "ward.D2")

# Evaluate silhouette for k = 2 to 10
sil_width_hc <- function(k) {
  clusters <- cutree(hc_res, k = k)
  mean(silhouette(clusters, dist_matrix)[, 3])
}

sil_scores_hc <- sapply(2:10, sil_width_hc)

# Plot silhouette scores
plot(2:10, sil_scores_hc, type = "b", 
     xlab = "Number of Clusters", ylab = "Average Silhouette Width",
     main = "Silhouette Method for Optimal K (Hierarchical)")

# Dendrogram
plot(hc_res, labels = FALSE, main = "Dendrogram (Ward's Method)")

# Cut into 9 clusters
hc_clusters <- cutree(hc_res, k = 9)
df_encoded$hc_cluster <- as.factor(hc_clusters)

# Visualization
fviz_cluster(list(data = features_scaled, cluster = hc_clusters), main = "Hierarchical Clusters (Ward)")

hc_stats <- cluster.stats(dist_matrix, hc_clusters)


# Davies-Bouldin Index
dbi <- intCriteria(
  as.matrix(features_scaled), 
  as.integer(kmeans_res$cluster), 
  c("Davies_Bouldin")
)

dbi_hc <- intCriteria(
  as.matrix(features_scaled),
  as.integer(hc_clusters),
  c("Davies_Bouldin")
)

cat("Davies-Bouldin Index (KMeans):", dbi$davies_bouldin, "\n")
cat("Davies-Bouldin Index (Hierarchical):", dbi_hc$davies_bouldin, "\n")

# Intra-cluster distance
intra_dist <- function(data, clusters) {
  mean(sapply(unique(clusters), function(k) {
    cl_data <- data[clusters == k, , drop = FALSE]
    if (nrow(cl_data) > 1) {
      mean(dist(cl_data))
    } else {
      0
    }
  }))
}

kmeans_intra_dist <- intra_dist(features_scaled, kmeans_res$cluster)
hc_intra_dist <- intra_dist(features_scaled, hc_clusters)

cat("Intra-cluster distance (KMeans):", kmeans_intra_dist, "\n")
cat("Intra-cluster distance (Hierarchical):", hc_intra_dist, "\n")

# ------------------------------
# FINAL COMPARISON & SELECTION
# ------------------------------

comparison_final <- tibble(
  Metric = c("Within-cluster SS", "Average Silhouette Width", 
             "Calinski-Harabasz Index", "Dunn Index", 
             "Davies-Bouldin Index", "Intra-cluster Distance"),
  KMeans = c(kmeans_stats$within.cluster.ss, kmeans_stats$avg.silwidth, 
             kmeans_stats$ch, kmeans_stats$dunn, 
             dbi$davies_bouldin, kmeans_intra_dist),
  Hierarchical = c(hc_stats$within.cluster.ss, hc_stats$avg.silwidth, 
                   hc_stats$ch, hc_stats$dunn, 
                   dbi_hc$davies_bouldin, hc_intra_dist)
)

cat("Final Comparison of Clustering Methods:\n")
comparison_final

# But in 2D, kmeans look more "separated"

df_encoded$kmeans_cluster <- as.factor(kmeans_res$cluster)
df_encoded$hc_cluster <- as.factor(hc_clusters)

df_clusters <- df_encoded %>%
  dplyr::select(annotator_id, kmeans_cluster, hc_cluster)

write.csv(df_clusters, "feature_files/features_task_4_train.csv", row.names = FALSE)


################## Dev and Test Dataset - Clusters


add_missing_columns <- function(df, reference_cols) {
  missing <- setdiff(reference_cols, names(df))
  for (col in missing) {
    df[[col]] <- 0
  }
  df <- df[, reference_cols]
  return(df)
}


# Process test/dev datasets
process_test_features <- function(df) {
  if ("annotator" %in% names(df)) {
    names(df)[names(df) == "annotator"] <- "annotator_id"
  }
  if ("age_group" %in% names(df)) {
    names(df)[names(df) == "age_group"] <- "age"
  }
  if ("study_level" %in% names(df)) {
    names(df)[names(df) == "study_level"] <- "education"
  }
  
  df$region <- countrycode(df$country, origin = 'country.name', destination = 'region')
  df$region[is.na(df$region)] <- "Other"
  
  demo_cols <- c("gender", "age", "education", "ethnicity", "region")
  available_demo_cols <- demo_cols[demo_cols %in% names(df)]
  
  df_encoded <- fastDummies::dummy_cols(df,
                                        select_columns = available_demo_cols,
                                        remove_first_dummy = TRUE,
                                        remove_selected_columns = TRUE)
  
  if ("annotator_id" %in% names(df)) {
    annotator_stats <- df %>%
      group_by(annotator_id) %>%
      summarise(n_labeled = n(), .groups = "drop")
    df_encoded <- df_encoded %>%
      left_join(annotator_stats, by = "annotator_id")
  } else {
    df_encoded$n_labeled <- NA
  }
  
  # These are NA in dev/test, will be removed before clustering
  df_encoded$sexist_rate <- NA
  df_encoded$agreement_rate <- NA
  
  return(df_encoded)
}


# Load processed training info
train_mean <- attr(features_scaled, "scaled:center")
train_sd   <- attr(features_scaled, "scaled:scale")
cluster_feature_names <- names(train_mean)

# Remove NA columns from clustering
cluster_feature_names_clean <- setdiff(cluster_feature_names, c("sexist_rate", "agreement_rate"))

# Load and preprocess dev/test
df_dev <- read.csv("data/EXIST2025_dev_labeled.csv")
df_test <- read.csv("data/EXIST_test_nolabel.csv")

df_dev_encoded <- process_test_features(df_dev)
df_test_encoded <- process_test_features(df_test)

# Match features to training
df_dev_fixed <- add_missing_columns(df_dev_encoded, cluster_feature_names)
df_test_fixed <- add_missing_columns(df_test_encoded, cluster_feature_names)

# Remove NA-feature columns
df_dev_fixed <- df_dev_fixed[, cluster_feature_names_clean]
df_test_fixed <- df_test_fixed[, cluster_feature_names_clean]

# Scale using training mean/sd
features_dev_scaled <- scale(df_dev_fixed,
                             center = train_mean[cluster_feature_names_clean],
                             scale = train_sd[cluster_feature_names_clean])

features_test_scaled <- scale(df_test_fixed,
                              center = train_mean[cluster_feature_names_clean],
                              scale = train_sd[cluster_feature_names_clean])

# NEW KMeans clustering (k = 4)
set.seed(42)
kmeans_dev <- kmeans(features_dev_scaled, centers = 4, nstart = 25)
kmeans_test <- kmeans(features_test_scaled, centers = 4, nstart = 25)

df_dev_encoded$kmeans_cluster <- as.factor(kmeans_dev$cluster)
df_test_encoded$kmeans_cluster <- as.factor(kmeans_test$cluster)

# Hierarchical clustering (k = 9)
hc_dev <- hclust(dist(features_dev_scaled), method = "ward.D2")
hc_test <- hclust(dist(features_test_scaled), method = "ward.D2")

df_dev_encoded$hc_cluster <- as.factor(cutree(hc_dev, k = 9))
df_test_encoded$hc_cluster <- as.factor(cutree(hc_test, k = 9))


# Save cluster assignments
df_clusters_dev <- df_dev_encoded %>%
  dplyr::select(annotator_id, kmeans_cluster, hc_cluster)

df_clusters_test <- df_test_encoded %>%
  dplyr::select(annotator_id, kmeans_cluster, hc_cluster)

write.csv(df_clusters_dev, "feature_files/features_task_4_dev.csv", row.names = FALSE)
write.csv(df_clusters_test, "feature_files/features_task_4_test.csv", row.names = FALSE)
