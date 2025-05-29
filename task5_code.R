# Task 5: Predicting Annotator Decisions (Enhanced Recommendation System Approach)
# ------------------------------------------------------------------------------
# Este código implementa uma abordagem de sistema de recomendação aprimorada para prever como 
# anotadores classificariam tweets com base no comportamento de anotadores similares.
# ------------------------------------------------------------------------------

# Carregar bibliotecas necessárias
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(recommenderlab)
library(reshape2)
library(caret)
library(rstatix)
library(viridis)
library(knitr)
library(parallel)
library(doParallel)  # Para processamento paralelo
library(corrplot)    # Para visualizar matrizes de correlação

# Configurar processamento paralelo para acelerar os cálculos
num_cores <- detectCores() - 1  # Deixar um núcleo livre para o sistema
if (num_cores > 1) {
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  print(paste("Usando", num_cores, "núcleos para processamento paralelo"))
}

# ------------------------------------------------------------------------------
# 1. Carregar e explorar os dados de treinamento
# ------------------------------------------------------------------------------
print("Carregando e explorando os dados de treinamento...")

# Carregar dados de treinamento
df_train <- read_csv("data/EXIST2025_train.csv")

# Exibir informações básicas sobre o conjunto de dados
print(paste("Número de linhas nos dados de treinamento:", nrow(df_train)))
print(paste("Número de tweets únicos:", length(unique(df_train$tweet))))
print(paste("Número de anotadores únicos:", length(unique(df_train$annotator_id))))

# Verificar a distribuição dos rótulos
label_distribution <- df_train %>%
  filter(!is.na(label_task1_1)) %>%
  group_by(label_task1_1) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

print("Distribuição dos rótulos:")
print(label_distribution)

# ------------------------------------------------------------------------------
# 2. Preprocessamento e limpeza de dados
# ------------------------------------------------------------------------------
print("Realizando preprocessamento e limpeza de dados...")

# 2.1 Identificar e lidar com anotadores inconsistentes
annotator_consistency <- df_train %>%
  filter(!is.na(label_task1_1)) %>%
  group_by(tweet) %>%
  mutate(
    majority_vote = ifelse(sum(label_task1_1 == "YES") > sum(label_task1_1 == "NO"), "YES", "NO")
  ) %>%
  ungroup() %>%
  group_by(annotator_id) %>%
  summarise(
    total_annotations = n(),
    agreement_with_majority = sum(label_task1_1 == majority_vote) / n(),
    yes_rate = sum(label_task1_1 == "YES") / n()
  )

# Visualizar distribuição de consistência dos anotadores
ggplot(annotator_consistency, aes(x = agreement_with_majority)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_vline(xintercept = median(annotator_consistency$agreement_with_majority), 
             color = "red", linetype = "dashed") +
  labs(title = "Distribuição da Consistência dos Anotadores",
       x = "Taxa de Concordância com Maioria",
       y = "Número de Anotadores") +
  theme_minimal()

# Identificar anotadores muito inconsistentes (outliers)
lower_threshold <- quantile(annotator_consistency$agreement_with_majority, 0.05)
print(paste("Limite inferior para consistência dos anotadores:", round(lower_threshold, 3)))

# Filtrar anotadores altamente inconsistentes
unreliable_annotators <- annotator_consistency %>%
  filter(agreement_with_majority < lower_threshold & total_annotations >= 5) %>%
  pull(annotator_id)

print(paste("Número de anotadores identificados como altamente inconsistentes:", 
            length(unreliable_annotators)))

# Opção 1: Remover anotadores inconsistentes
# df_clean <- df_train %>% filter(!(annotator_id %in% unreliable_annotators))

# Opção 2: Ponderar anotadores (será implementado no sistema de recomendação)
df_clean <- df_train

# Verificar tweets sem anotações suficientes após filtragem
min_annotations_per_tweet <- 3
tweets_with_few_annotations <- df_clean %>%
  group_by(tweet) %>%
  summarise(num_annotations = sum(!is.na(label_task1_1))) %>%
  filter(num_annotations < min_annotations_per_tweet) %>%
  pull(tweet)

print(paste("Número de tweets com menos de", min_annotations_per_tweet, "anotações:", 
            length(tweets_with_few_annotations)))

# 2.2 Preparação de dados para o sistema de recomendação
print("Preparando dados para o sistema de recomendação...")

# Converter rótulos para formato numérico (2 para SIM, 1 para NÃO)
df_reco <- df_clean %>%
  select(annotator_id, tweet, label_task1_1) %>%
  mutate(label_task1_1 = ifelse(label_task1_1 == "YES", 2,
                                ifelse(label_task1_1 == "NO", 1, NA)))

# Criar uma matriz usuário-item para o sistema de recomendação
rating_matrix <- dcast(df_reco, annotator_id ~ tweet, value.var = "label_task1_1")

# Verificar dimensões da matriz de classificação
print(paste("Dimensões da matriz de classificação:", nrow(rating_matrix), "x", ncol(rating_matrix) - 1))

# Converter para o formato de matriz exigido pelo recommenderlab
mat <- as.matrix(rating_matrix[,-1])
rownames(mat) <- rating_matrix$annotator_id

# 2.3 Calcular e armazenar pesos de confiabilidade dos anotadores
annotator_weights <- annotator_consistency %>%
  select(annotator_id, agreement_with_majority) %>%
  rename(weight = agreement_with_majority)

# Normalizar pesos para estar entre 0.5 e 1
annotator_weights <- annotator_weights %>%
  mutate(weight = 0.5 + 0.5 * (weight - min(weight)) / (max(weight) - min(weight)))

# Analisar tweets por anotador (necessário para o parâmetro 'given')
tweets_per_annotator <- df_train %>%
  group_by(annotator_id) %>%
  summarise(tweets_rated = sum(!is.na(label_task1_1)))

# ------------------------------------------------------------------------------
# 3. Criar várias versões da matriz de rating com diferentes normalizações
# ------------------------------------------------------------------------------
print("Criando diferentes versões da matriz de rating...")

# 3.1 Matriz original (sem normalização)
rrm_original <- as(mat, "realRatingMatrix")

# 3.2 Matriz com normalização por centro
# Calcular médias por linha ignorando NAs
row_means <- rowMeans(mat, na.rm = TRUE)
# Substituir NaN (quando toda linha é NA) por 0
row_means[is.nan(row_means)] <- 0
# Centralizar cada linha
mat_centered <- mat - row_means
# Converter para realRatingMatrix
rrm_centered <- as(mat_centered, "realRatingMatrix")

# 3.3 Matriz com normalização Z-score
# Calcular desvios padrão por linha ignorando NAs
row_sds <- apply(mat, 1, sd, na.rm = TRUE)
# Substituir 0 ou NaN por 1 para evitar divisão por zero
row_sds[is.nan(row_sds) | row_sds == 0] <- 1
# Aplicar Z-score
mat_zscore <- sweep(mat_centered, 1, row_sds, "/")
# Converter para realRatingMatrix
rrm_zscore <- as(mat_zscore, "realRatingMatrix")

# 3.4 Matriz com normalização Min-Max
min_max_normalize <- function(x) {
  min_val <- min(x, na.rm = TRUE)
  max_val <- max(x, na.rm = TRUE)
  # Verificar se min e max são iguais (todos valores iguais)
  if (min_val == max_val) {
    return(rep(0.5, length(x)))
  } else {
    return((x - min_val) / (max_val - min_val))
  }
}

# Aplicar normalização min-max por linha
mat_minmax <- mat
for (i in 1:nrow(mat)) {
  if (sum(!is.na(mat[i,])) > 1) {  # Se houver pelo menos 2 valores não-NA
    row_values <- mat[i,]
    normalized <- min_max_normalize(row_values)
    mat_minmax[i, !is.na(row_values)] <- normalized[!is.na(row_values)]
  }
}

# Converter para escala de 1 a 2 (NO a YES)
mat_minmax <- 1 + mat_minmax
# Converter para realRatingMatrix
rrm_minmax <- as(mat_minmax, "realRatingMatrix")

# Lista de matrizes para testar
rating_matrices <- list(
  "original" = rrm_original,
  "centered" = rrm_centered,
  "zscore" = rrm_zscore,
  "minmax" = rrm_minmax
)

# ------------------------------------------------------------------------------
# 4. Configurar parâmetros para ajuste de hiperparâmetros
# ------------------------------------------------------------------------------
print("Configurando parâmetros para ajuste de hiperparâmetros...")

# Definir valores de hiperparâmetros para testar
params <- list(
  UBCF = list(
    nn = c(3, 5, 7, 10, 15, 20, 30, 50),
    method = c("cosine", "pearson", "jaccard")
  ),
  IBCF = list(
    k = c(5, 10, 15, 20, 30, 50, 100),
    method = c("cosine", "pearson", "jaccard")
  ),
  SVD = list(
    k = c(5, 10, 20, 50),
    normalize = c(TRUE, FALSE)
  ),
  SVDF = list(
    k = c(5, 10, 20, 50),
    normalize = c("center", "Z-score", "none")
  ),
  HYBRID = list(
    weights = list(
      c(0.4, 0.3, 0.3),  # POPULAR, UBCF, IBCF
      c(0.3, 0.4, 0.3),
      c(0.3, 0.3, 0.4),
      c(0.5, 0.25, 0.25),
      c(0.25, 0.5, 0.25),
      c(0.25, 0.25, 0.5),
      c(0.6, 0.2, 0.2),
      c(0.2, 0.6, 0.2),
      c(0.2, 0.2, 0.6)
    )
  )
)

# ------------------------------------------------------------------------------
# 5. Definir função de avaliação com métricas para classificação binária
# ------------------------------------------------------------------------------
print("Definindo função de avaliação do modelo com métricas adaptadas...")

evaluate_model <- function(e_scheme, method_name, params = NULL, verbose = TRUE) {
  if (verbose) {
    cat("\n---", method_name, "---\n")
    if (!is.null(params)) {
      cat("Parâmetros:", paste(names(params), params, sep = "=", collapse = ", "), "\n")
    }
  }
  
  # Treinar o modelo
  tryCatch({
    model <- Recommender(getData(e_scheme, "train"), method = method_name, parameter = params)
    
    # Avaliar o modelo (para ratings)
    res <- evaluate(e_scheme, method = method_name, type = "ratings", parameter = params)
    
    # Extrair e imprimir métricas de avaliação
    metrics <- avg(res)
    if (verbose) {
      print(metrics)
    }
    
    # Converter métricas para data frame
    metrics_df <- as.data.frame(metrics)
    colnames(metrics_df) <- c("RMSE", "MSE", "MAE")
    
    # Também avaliar como classificação binária (para adequar às métricas do trabalho)
    # Prever no conjunto de teste
    predictions <- predict(model, getData(e_scheme, "known"), type = "ratings")
    
    # Converter para matriz e normalizar para binário (>= 1.5 = "YES", < 1.5 = "NO")
    pred_matrix <- as(predictions, "matrix")
    actual_matrix <- as(getData(e_scheme, "unknown"), "matrix")
    
    # Aplacenar os valores
    pred_flat <- as.vector(pred_matrix)
    actual_flat <- as.vector(actual_matrix)
    
    # Remover NAs
    valid_idx <- which(!is.na(pred_flat) & !is.na(actual_flat))
    pred_clean <- pred_flat[valid_idx]
    actual_clean <- actual_flat[valid_idx]
    
    # Converter para classificação binária
    pred_binary <- ifelse(pred_clean >= 1.5, 2, 1)
    
    # Calcular métricas de classificação
    if(length(pred_binary) > 0) {
      confusion <- confusionMatrix(factor(pred_binary), factor(actual_clean))
      
      # Adicionar métricas de classificação ao data frame
      class_metrics <- data.frame(
        Accuracy = confusion$overall["Accuracy"],
        F1_Score = confusion$byClass["F1"],
        Precision = confusion$byClass["Precision"],
        Recall = confusion$byClass["Recall"],
        Kappa = confusion$overall["Kappa"]
      )
      
      # Combinar métricas
      all_metrics <- cbind(metrics_df, class_metrics)
    } else {
      all_metrics <- metrics_df
      all_metrics$Accuracy <- NA
      all_metrics$F1_Score <- NA
      all_metrics$Precision <- NA
      all_metrics$Recall <- NA
      all_metrics$Kappa <- NA
    }
    
    if (verbose) {
      # Plotar métricas de avaliação
      par(mfrow = c(1, 2))
      
      # Métricas de recomendação
      barplot(as.matrix(metrics_df), 
              beside = TRUE, 
              col = viridis(3),
              main = paste("Métricas de Recomendação para", method_name),
              ylim = c(0, max(metrics_df, na.rm = TRUE) * 1.2),
              ylab = "Valor")
      legend("topright", 
             legend = colnames(metrics_df),
             fill = viridis(3),
             cex = 0.8)
      
      # Métricas de classificação
      if(!all(is.na(all_metrics[,4:8]))) {
        barplot(as.matrix(all_metrics[,4:8]), 
                beside = TRUE, 
                col = viridis(5),
                main = paste("Métricas de Classificação para", method_name),
                ylim = c(0, 1),
                ylab = "Valor")
        legend("bottomright", 
               legend = colnames(all_metrics)[4:8],
               fill = viridis(5),
               cex = 0.8)
      }
      
      # Restaurar layout
      par(mfrow = c(1, 1))
      
      # Plotar distribuição de erro (para recomendação)
      plot(res, main = paste("Distribuição de Erros para", method_name))
    }
    
    return(list(model = model, metrics = all_metrics, params = params))
  }, error = function(e) {
    cat("Erro ao avaliar", method_name, "com parâmetros", paste(names(params), params, sep = "=", collapse = ", "), ":\n")
    cat("  ", conditionMessage(e), "\n")
    return(NULL)
  })
}

# Função para avaliar modelo híbrido (requer implementação especial)
evaluate_hybrid_model <- function(e_scheme, weight_params) {
  method_name <- paste("HYBRID", 
                       paste(round(weight_params * 100), collapse = "-"), 
                       sep = "_")
  
  cat("\n---", method_name, "---\n")
  cat("Pesos:", paste(round(weight_params, 2), collapse = ", "), "\n")
  
  # Obter dados de treinamento
  train_data <- getData(e_scheme, "train")
  
  # Treinar modelos componentes
  model_popular <- Recommender(train_data, method = "POPULAR")
  model_ubcf <- Recommender(train_data, method = "UBCF", 
                            parameter = list(method = "cosine", nn = 20))
  model_ibcf <- Recommender(train_data, method = "IBCF", 
                            parameter = list(method = "cosine", k = 30))
  
  # Função para gerar previsões híbridas
  make_hybrid_predictions <- function(test_data) {
    # Obter previsões de cada modelo
    pred_popular <- predict(model_popular, test_data, type = "ratings")
    pred_ubcf <- predict(model_ubcf, test_data, type = "ratings")
    pred_ibcf <- predict(model_ibcf, test_data, type = "ratings")
    
    # Converter para matrizes
    mat_popular <- as(pred_popular, "matrix")
    mat_ubcf <- as(pred_ubcf, "matrix")
    mat_ibcf <- as(pred_ibcf, "matrix")
    
    # Combinar com pesos
    combined_pred <- weight_params[1] * mat_popular + 
      weight_params[2] * mat_ubcf + 
      weight_params[3] * mat_ibcf
    
    # Converter de volta para realRatingMatrix
    return(as(combined_pred, "realRatingMatrix"))
  }
  
  # Fazer previsões no conjunto de teste
  known_ratings <- getData(e_scheme, "known")
  unknown_ratings <- getData(e_scheme, "unknown")
  
  hybrid_predictions <- make_hybrid_predictions(known_ratings)
  
  # Calcular métricas para ratings
  pred_matrix <- as(hybrid_predictions, "matrix")
  actual_matrix <- as(unknown_ratings, "matrix")
  
  # Calcular RMSE, MSE, MAE
  errors <- pred_matrix - actual_matrix
  valid_errors <- errors[!is.na(errors)]
  
  rmse <- sqrt(mean(valid_errors^2))
  mse <- mean(valid_errors^2)
  mae <- mean(abs(valid_errors))
  
  metrics_df <- data.frame(RMSE = rmse, MSE = mse, MAE = mae)
  
  # Calcular métricas de classificação binária
  pred_flat <- as.vector(pred_matrix)
  actual_flat <- as.vector(actual_matrix)
  
  # Remover NAs
  valid_idx <- which(!is.na(pred_flat) & !is.na(actual_flat))
  pred_clean <- pred_flat[valid_idx]
  actual_clean <- actual_flat[valid_idx]
  
  # Converter para classificação binária
  pred_binary <- ifelse(pred_clean >= 1.5, 2, 1)
  
  # Calcular métricas de classificação
  if(length(pred_binary) > 0) {
    confusion <- confusionMatrix(factor(pred_binary), factor(actual_clean))
    
    # Adicionar métricas de classificação ao data frame
    class_metrics <- data.frame(
      Accuracy = confusion$overall["Accuracy"],
      F1_Score = confusion$byClass["F1"],
      Precision = confusion$byClass["Precision"],
      Recall = confusion$byClass["Recall"],
      Kappa = confusion$overall["Kappa"]
    )
    
    # Combinar métricas
    all_metrics <- cbind(metrics_df, class_metrics)
  } else {
    all_metrics <- metrics_df
    all_metrics$Accuracy <- NA
    all_metrics$F1_Score <- NA
    all_metrics$Precision <- NA
    all_metrics$Recall <- NA
    all_metrics$Kappa <- NA
  }
  
  print(all_metrics)
  
  # Plotar métricas de avaliação
  par(mfrow = c(1, 2))
  
  # Métricas de recomendação
  barplot(as.matrix(metrics_df), 
          beside = TRUE, 
          col = viridis(3),
          main = paste("Métricas de Recomendação para", method_name),
          ylim = c(0, max(metrics_df) * 1.2),
          ylab = "Valor")
  legend("topright", 
         legend = colnames(metrics_df),
         fill = viridis(3),
         cex = 0.8)
  
  # Métricas de classificação
  barplot(as.matrix(all_metrics[,4:8]), 
          beside = TRUE, 
          col = viridis(5),
          main = paste("Métricas de Classificação para", method_name),
          ylim = c(0, 1),
          ylab = "Valor")
  legend("bottomright", 
         legend = colnames(all_metrics)[4:8],
         fill = viridis(5),
         cex = 0.8)
  
  # Restaurar layout
  par(mfrow = c(1, 1))
  
  # Criar um modelo híbrido como objeto para retornar
  hybrid_model <- list(
    models = list(popular = model_popular, ubcf = model_ubcf, ibcf = model_ibcf),
    weights = weight_params,
    predict = make_hybrid_predictions
  )
  
  return(list(model = hybrid_model, metrics = all_metrics, params = weight_params))
}

# ------------------------------------------------------------------------------
# 6. Executar experimentos com validação cruzada
# ------------------------------------------------------------------------------
print("Iniciando experimentos com validação cruzada robusta...")

# Resultados para armazenar os resultados de todos os experimentos
all_results <- list()

# Analisar tweets por anotador (necessário para o parâmetro 'given')
tweets_per_annotator <- df_train %>%
  group_by(annotator_id) %>%
  summarise(tweets_rated = sum(!is.na(label_task1_1)))

# Para cada tipo de matriz de avaliação
for (matrix_name in names(rating_matrices)) {
  print(paste("Avaliando com matriz:", matrix_name))
  rrm <- rating_matrices[[matrix_name]]
  
  # Configurar esquema de validação cruzada
  k_folds <- 5  # Número de folds para validação cruzada
  
  # Determinar o parâmetro 'given' com base na análise de dados
  median_tweets_rated <- median(tweets_per_annotator$tweets_rated)
  given_param <- floor(median_tweets_rated * 0.7)  # Usar 70% para treinamento
  
  # Criar esquema de avaliação
  e_scheme <- evaluationScheme(rrm, method = "cross-validation", 
                               k = k_folds,  # k-fold cross validation
                               given = given_param, 
                               goodRating = 2)  # 2 = tweet sexista
  
  print(paste("Usando validação cruzada", k_folds, "folds com given =", given_param))
  
  # 6.1 Modelo de linha de base (RANDOM)
  all_results[[paste(matrix_name, "RANDOM", sep = "_")]] <- 
    evaluate_model(e_scheme, "RANDOM")
  
  # 6.2 Modelo POPULAR
  all_results[[paste(matrix_name, "POPULAR", sep = "_")]] <- 
    evaluate_model(e_scheme, "POPULAR")
  
  # 6.3 Filtragem colaborativa baseada em usuário (UBCF)
  for (nn in params$UBCF$nn) {
    for (method in params$UBCF$method) {
      result_key <- paste(matrix_name, paste0("UBCF_", method, "_nn", nn), sep = "_")
      all_results[[result_key]] <- evaluate_model(
        e_scheme, "UBCF", 
        params = list(method = method, nn = nn),
        verbose = FALSE  # Reduzir saída para muitos experimentos
      )
    }
  }
  
  # 6.4 Filtragem colaborativa baseada em item (IBCF)
  for (k in params$IBCF$k) {
    for (method in params$IBCF$method) {
      result_key <- paste(matrix_name, paste0("IBCF_", method, "_k", k), sep = "_")
      all_results[[result_key]] <- evaluate_model(
        e_scheme, "IBCF", 
        params = list(method = method, k = k),
        verbose = FALSE
      )
    }
  }
  
  # 6.5 Decomposição em valores singulares (SVD)
  for (k in params$SVD$k) {
    for (normalize in params$SVD$normalize) {
      result_key <- paste(matrix_name, paste0("SVD_k", k, "_norm", normalize), sep = "_")
      all_results[[result_key]] <- evaluate_model(
        e_scheme, "SVD", 
        params = list(k = k, normalize = normalize),
        verbose = FALSE
      )
    }
  }
  
  # 6.6 Fatorial SVD (SVDF)
  for (k in params$SVDF$k) {
    for (normalize in params$SVDF$normalize) {
      result_key <- paste(matrix_name, paste0("SVDF_k", k, "_norm", normalize), sep = "_")
      all_results[[result_key]] <- evaluate_model(
        e_scheme, "SVDF", 
        params = list(k = k, normalize = normalize),
        verbose = FALSE
      )
    }
  }
  
  # 6.7 Modelos híbridos
  for (weight_set in params$HYBRID$weights) {
    result_key <- paste(matrix_name, paste0("HYBRID_", 
                                            paste(round(weight_set * 100), collapse = "-")), 
                        sep = "_")
    all_results[[result_key]] <- evaluate_hybrid_model(e_scheme, weight_set)
  }
}

# ------------------------------------------------------------------------------
# 7. Analisar resultados e selecionar o melhor modelo
# ------------------------------------------------------------------------------
print("Analisando resultados e selecionando o melhor modelo...")

# Extrair métricas de todos os modelos para um dataframe
results_df <- data.frame()

extract_metrics_from_result <- function(result, experiment_name) {
  if(is.null(result) || is.null(result$metrics)) {
    return(NULL)
  }
  
  metrics <- result$metrics
  metrics$Experiment <- experiment_name
  
  # Adicionar informações do modelo
  if(!is.null(result$params)) {
    param_names <- names(result$params)
    for(param in param_names) {
      param_value <- result$params[[param]]
      if(length(param_value) == 1) {
        metrics[[param]] <- param_value
      } else if(is.character(param_value) || is.numeric(param_value)) {
        metrics[[param]] <- paste(param_value, collapse = ",")
      }
    }
  }
  
  return(metrics)
}

# Extrair métricas de todos os experimentos
for(experiment_name in names(all_results)) {
  result <- all_results[[experiment_name]]
  metrics_df <- extract_metrics_from_result(result, experiment_name)
  
  if(!is.null(metrics_df)) {
    results_df <- rbind(results_df, metrics_df)
  }
}

# Ordenar resultados por F1-Score (ou outra métrica importante)
if("F1_Score" %in% colnames(results_df)) {
  results_df <- results_df[order(-results_df$F1_Score), ]
} else if("RMSE" %in% colnames(results_df)) {
  results_df <- results_df[order(results_df$RMSE), ]
}

# Mostrar os melhores modelos
print("Top 10 modelos por F1-Score (ou RMSE):")
top_models <- head(results_df, 10)
print(kable(top_models))

# Visualize comparison of top models
if (nrow(top_models) > 0) {
  # Prepare data for visualization
  top_models_viz <- top_models[, c("Experiment", "RMSE", "Accuracy", "F1_Score", "Precision", "Recall")]
  
  # Convert to long format for ggplot
  top_models_long <- reshape2::melt(top_models_viz, id.vars = "Experiment", 
                                    variable.name = "Metric", value.name = "Value")
  
  # Plot comparison
  ggplot(top_models_long, aes(x = reorder(Experiment, -Value), y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~ Metric, scales = "free_y") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(
      title = "Comparison of Top Models by Metric",
      x = "Experiment",
      y = "Metric Value",
      fill = "Metric"
    )
}

