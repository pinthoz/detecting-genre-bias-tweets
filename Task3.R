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

