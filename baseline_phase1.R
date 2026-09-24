
install.packages("caret")
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(dplyr))
# SETWD
#===============================================================================

# Check the current working directory
current_directory <- getwd()

# Print the current working directory
print(current_directory)

# Change working directory
setwd("/Users/...")

# Check the current working directory
current_directory <- getwd()

# Print the current working directory
print(current_directory)

# READ
#===============================================================================

# Read the CSV file
train_synthetic <- read.csv("train-synthetic_over.csv")
train_synthetic_shuffled <- train_synthetic[sample(nrow(train_synthetic)),]

test_synthetic <- read.csv("test-synthetic_over.csv")
test_synthetic_shuffled <- test_synthetic[sample(nrow(test_synthetic)),]


#================

# Set up splits
features_synthetic <- train_synthetic_shuffled %>% dplyr::select(-Label, -NextAdmissionDays, -VISIT_COUNT, -X)
target_synthetic <- as.factor(train_synthetic_shuffled$Label)
levels(target_synthetic) <- c("No_Readmission", "Readmission")  

features_synthetic_test <- test_synthetic_shuffled %>% dplyr::select(-Label, -NextAdmissionDays, -VISIT_COUNT, -X)
target_synthetic_test <- as.factor(test_synthetic_shuffled$Label)
levels(target_synthetic_test) <- c("No_Readmission", "Readmission")  

write.csv(train_synthetic_shuffled, 'train-synthetic_shuffled.csv')
write.csv(test_synthetic_shuffled, 'test-synthetic_shuffled.csv')



# Set up training control
train_control <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 3, 
  summaryFunction = twoClassSummary, 
  classProbs = TRUE, # for ROC curve
  sampling = "down" # "up" or "smote" for handling imbalance
)

# Train Random Forest
set.seed(123)
rf_model_synthetic <- train(
  x = features_synthetic, 
  y = target_synthetic, 
  method = "rf", 
  trControl = train_control, 
  metric = "ROC"
)

#-------------------------------- Scoring for the full test set

pred_synthetic_test <- predict(rf_model_synthetic, features_synthetic_test)
pred_proba_synthetic_test <- predict(rf_model_synthetic, features_synthetic_test, type="prob")

write.csv(pred_proba_synthetic_test, 'pred_proba_synthetic_test.csv')
write.csv(pred_synthetic_test, 'pred_synthetic_test.csv')

conf_matrix_synthetic <- confusionMatrix(pred_synthetic_test, target_synthetic_test)


# Extracting the elements of the confusion matrix
tp_synthetic <- as.numeric(conf_matrix_synthetic$table[1, 1])
tn_synthetic <- as.numeric(conf_matrix_synthetic$table[2, 2])
fp_synthetic <- as.numeric(conf_matrix_synthetic$table[1, 2])
fn_synthetic <- as.numeric(conf_matrix_synthetic$table[2, 1])

# Manual calculation of MCC
mcc_synthetic <- (tp_synthetic * tn_synthetic - fp_synthetic * fn_synthetic) / sqrt((tp_synthetic + fp_synthetic) * (tp_synthetic + fn_synthetic) * (tn_synthetic + fp_synthetic) * (tn_synthetic + fn_synthetic))

print(mcc_synthetic)


#-------------------------------- Double check for scoring 
# Calculate Precision and Recall
precision_synthetic <- tp_synthetic / (tp_synthetic + fp_synthetic)
recall_synthetic <- tp_synthetic / (tp_synthetic + fn_synthetic)
print(precision_synthetic)
print(recall_synthetic)

# Calculate F1 Score
f1_score_synthetic <- 2 * ((precision_synthetic * recall_synthetic) / (precision_synthetic + recall_synthetic))
print(f1_score_synthetic)


# Sensitivity (Recall) for the positive class
sensitivity <- conf_matrix_synthetic$byClass["Sensitivity"]

# Specificity for the negative class
specificity <- conf_matrix_synthetic$byClass["Specificity"]

# Precision (Positive Predictive Value)
precision <- conf_matrix_synthetic$byClass["Pos Pred Value"]

# Print extracted metrics
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("Precision:", precision))

# Calculate F1 Score
f1_score <- 2 * ((precision * sensitivity) / (precision + sensitivity))

# Print F1 score
print(paste("F1 Score:", f1_score))

#===============================================================================
# REAL-WORLD UTILITY: APPLY SYNTHETIC-TRAINED RF TO ORIGINAL DATA
#===============================================================================

suppressPackageStartupMessages(library(FNN))
suppressPackageStartupMessages(library(pROC))

# Load original datasets
train_original <- read.csv("train-original.csv")
test_original  <- read.csv("test-original.csv")

# Use exactly the predictors used by the synthetic RF model
rf_predictor_names <- names(features_synthetic)

# Check whether the original datasets contain all required predictors
missing_train_columns <- setdiff(rf_predictor_names, names(train_original))
missing_test_columns  <- setdiff(rf_predictor_names, names(test_original))

if (length(missing_train_columns) > 0) {
  stop("Original training data is missing these predictors: ",
    paste(missing_train_columns, collapse = ", "))}

if (length(missing_test_columns) > 0) {
  stop("Original test data is missing these predictors: ",
    paste(missing_test_columns, collapse = ", "))}

# Retain the same predictors and the same column order
features_original_train <- train_original[, rf_predictor_names, drop = FALSE]

features_original_test <- test_original[, rf_predictor_names, drop = FALSE]


# Match original-data variable types to synthetic training data
#------------------------

align_to_synthetic <- function(original_features, synthetic_features, dataset_name) {
  aligned <- original_features[, names(synthetic_features), drop = FALSE]
  
  for (column_name in names(synthetic_features)) {
    synthetic_column <- synthetic_features[[column_name]]
    original_column  <- aligned[[column_name]]
    # Match factor variables
    if (is.factor(synthetic_column)) {
      original_values <- as.character(original_column)
      unseen_levels <- setdiff(unique(original_values[!is.na(original_values)]),
        levels(synthetic_column))
      
      if (length(unseen_levels) > 0) {
        stop(dataset_name, " contains unseen levels in predictor '",
          column_name, "': ", paste(unseen_levels, collapse = ", "))
      }
      
      aligned[[column_name]] <- factor(
        original_values,
        levels = levels(synthetic_column))
      
      # Match character variables
    } else if (is.character(synthetic_column)) {
      synthetic_levels <- unique(synthetic_column)
      original_values <- as.character(original_column)
      unseen_levels <- setdiff(unique(original_values[!is.na(original_values)]),
        synthetic_levels)
      
      if (length(unseen_levels) > 0) {
        stop(dataset_name, " contains unseen values in predictor '",
          column_name, "': ", paste(unseen_levels, collapse = ", "))
          }
      
      aligned[[column_name]] <- original_values
      
      # Match numeric variables
    } else if (is.numeric(synthetic_column) || is.integer(synthetic_column)) {
      converted_values <- suppressWarnings(
        as.numeric(as.character(original_column)))
      
      conversion_problem <- (is.na(converted_values) & !is.na(original_column))
      
      if (any(conversion_problem)) {
        stop(dataset_name, " contains non-numeric values in numeric predictor '",
          column_name, "'.")}
      
      aligned[[column_name]] <- converted_values
      }
    }
  
  if (anyNA(aligned)) {
    na_columns <- names(aligned)[ colSums(is.na(aligned)) > 0]
    
    stop( dataset_name, " contains missing values after alignment in: ",
      paste(na_columns, collapse = ", "))
  }
  aligned
}

# Replace unseen categorical values with the synthetic-training mode
#-----------------------
replace_unseen_with_mode <- function(original_features,
                                     synthetic_features,
                                     dataset_name) {
  
  replacement_log <- list()
  for (column_name in names(synthetic_features)) {
    synthetic_column <- synthetic_features[[column_name]]
    # Only categorical predictors can contain unseen levels
    if (is.factor(synthetic_column) || is.character(synthetic_column)) {
      
      synthetic_values <- as.character(synthetic_column)
      original_values  <- as.character(original_features[[column_name]])
      
      # Values actually observed during synthetic training
      allowed_values <- unique(synthetic_values[!is.na(synthetic_values) & synthetic_values != ""])
      if (length(allowed_values) == 0) {
        stop("Cannot calculate the mode for predictor '", column_name,
          "' because the synthetic training column is empty.")
      }
      
      # Most common observed synthetic-training value
      value_counts <- table(synthetic_values[!is.na(synthetic_values) & synthetic_values != ""])
      most_common_value <- names(value_counts)[which.max(value_counts)]
      
      # Identify values not observed during synthetic training
      unseen_rows <- (!is.na(original_values) & original_values != "" &
          !(original_values %in% allowed_values))

      if (any(unseen_rows)) {
        unseen_values <- sort(unique(original_values[unseen_rows]))
        
        number_replaced <- sum(unseen_rows)
        message(dataset_name, ": replacing ", number_replaced, " unseen value(s) in '",
          column_name, "' [", paste(unseen_values, collapse = ", "), "] with mode '",
          most_common_value, "'.")
        
        replacement_log[[length(replacement_log) + 1]] <-
          data.frame(
            Dataset = dataset_name,
            Predictor = column_name,
            Unseen_Values = paste(
              unseen_values,
              collapse = "; "),
            Replacement = most_common_value,
            Number_Replaced = number_replaced,
            stringsAsFactors = FALSE)
        
        original_values[unseen_rows] <- most_common_value
      }
      
      # Preserve the predictor type used by the model
      if (is.factor(synthetic_column)) {
        original_features[[column_name]] <- factor(original_values,
                                                  levels = levels(synthetic_column))
      } else {
        original_features[[column_name]] <- original_values
      }
    }
  }
  
  attr(original_features, "replacement_log") <- dplyr::bind_rows(replacement_log)
  original_features
}

# Replace unseen values in the original training data
features_original_train <- replace_unseen_with_mode(
  original_features = features_original_train,
  synthetic_features = features_synthetic,
  dataset_name = "Original training data")

original_train_replacement_log <- attr(features_original_train, "replacement_log")

# Replace unseen values in the original test data
features_original_test <- replace_unseen_with_mode(
  original_features = features_original_test,
  synthetic_features = features_synthetic,
  dataset_name = "Original test data")

original_test_replacement_log <- attr(features_original_test, "replacement_log")

# Replacements output
unseen_value_replacement_log <- dplyr::bind_rows(
  original_train_replacement_log,
  original_test_replacement_log)

print(unseen_value_replacement_log)

write.csv(unseen_value_replacement_log,
  "unseen_value_replacement_log.csv", row.names = FALSE)

#-------------------------------------------------------------------------------
# Now perform the normal alignment
#-------------------------------------------------------------------------------

features_original_train <- align_to_synthetic(features_original_train,
  features_synthetic, "Original training data")

features_original_test <- align_to_synthetic(features_original_test,
  features_synthetic, "Original test data")

features_original_train <- align_to_synthetic(features_original_train,
  features_synthetic, "Original training data")

features_original_test <- align_to_synthetic(features_original_test,
  features_synthetic, "Original test data")

# Column names and order is identical 
stopifnot(identical(names(features_original_train),names(features_synthetic)))

stopifnot(identical(names(features_original_test),names(features_synthetic)))

#-------------------------------------------------------------------------------
# Prepare original outcome labels
#-------------------------------------------------------------------------------

prepare_original_label <- function(x) {
  x <- trimws(as.character(x))
  
  converted <- dplyr::case_when(
    x %in% c("0", "No_Readmission", "No Readmission", "No") ~
      "No_Readmission",
    x %in% c("1", "Readmission", "Yes") ~
      "Readmission",
    TRUE ~ NA_character_)
  
  if (anyNA(converted)) {
    unknown_values <- unique(x[is.na(converted)])
    stop("Unknown original Label values: ", paste(unknown_values, collapse = ", "))
  }
  
  factor(converted, levels = levels(target_synthetic))
}

target_original_train <- prepare_original_label(train_original$Label)
target_original_test <- prepare_original_label(test_original$Label)

#-------------------------------------------------------------------------------
# Function to evaluate real-world utility
#-------------------------------------------------------------------------------

evaluate_real_world_utility <- function(model,
                                        features,
                                        actual,
                                        dataset_name) {
  
  predicted_class <- predict(model, newdata = features, type = "raw")
  
  predicted_probability <- predict(model, newdata = features, type = "prob")
  
  # Explicitly treat readmission as the clinically relevant positive class
  confusion <- confusionMatrix(data = predicted_class, reference = actual,
                                positive = "Readmission")
  
  # Rows - predictions and columns - reference values
  tp <- as.numeric(confusion$table["Readmission", "Readmission"])
  tn <- as.numeric(confusion$table["No_Readmission", "No_Readmission"])
  fp <- as.numeric(confusion$table["Readmission", "No_Readmission"])
  fn <- as.numeric(confusion$table["No_Readmission", "Readmission"])
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  sensitivity <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
  specificity <- ifelse(tn + fp == 0, NA, tn / (tn + fp))
  precision <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
  negative_predictive_value <- ifelse(tn + fn == 0, NA, tn / (tn + fn))
  
  f1_score <- ifelse(is.na(precision) || is.na(sensitivity) || 
                precision + sensitivity == 0, NA, 2 * precision * sensitivity /
                                                    (precision + sensitivity))
  
  balanced_accuracy <- mean(c(sensitivity, specificity), na.rm = TRUE)
  mcc_denominator <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  
  mcc <- ifelse(mcc_denominator == 0, NA, (tp * tn - fp * fn) / mcc_denominator)
  
  roc_result <- pROC::roc(
    response = actual,
    predictor = predicted_probability$Readmission,
    levels = c("No_Readmission", "Readmission"),
    direction = "<",
    quiet = TRUE)
  
  roc_auc <- as.numeric(pROC::auc(roc_result))
  # Brier score: lower values indicate better probability predictions
  observed_binary <- ifelse(actual == "Readmission", 1, 0)
  brier_score <- mean((predicted_probability$Readmission - observed_binary)^2)
  prevalence <- mean(actual == "Readmission")
  no_information_rate <- max(mean(actual == "Readmission"), 
                              mean(actual == "No_Readmission"))
  
  metrics <- data.frame(
    Dataset = dataset_name,
    N = length(actual),
    Readmission_Prevalence = prevalence,
    Accuracy = accuracy,
    No_Information_Rate = no_information_rate,
    Balanced_Accuracy = balanced_accuracy,
    Sensitivity_Recall = sensitivity,
    Specificity = specificity,
    Precision_PPV = precision,
    Negative_Predictive_Value = negative_predictive_value,
    F1_Score = f1_score,
    MCC = mcc,
    ROC_AUC = roc_auc,
    Brier_Score = brier_score
  )
  
  prediction_results <- data.frame(
    Actual = actual,
    Predicted = predicted_class,
    Probability_No_Readmission =
      predicted_probability$No_Readmission,
    Probability_Readmission =
      predicted_probability$Readmission
  )

  cat("\n===============================================\n")
  cat("Real-world evaluation:", dataset_name, "\n")
  cat("=================================================\n")
  print(confusion$table)
  print(metrics)
  list(
    predictions = prediction_results,
    probabilities = predicted_probability,
    confusion_matrix = confusion,
    metrics = metrics,
    roc = roc_result)
}

#-------------------------------------------------------------------------------
# Evaluate the synthetic-trained model on original train and test data
#-------------------------------------------------------------------------------

original_train_evaluation <- evaluate_real_world_utility(
  model = rf_model_synthetic,
  features = features_original_train,
  actual = target_original_train,
  dataset_name = "Original train"
)

original_test_evaluation <- evaluate_real_world_utility(
  model = rf_model_synthetic,
  features = features_original_test,
  actual = target_original_test,
  dataset_name = "Original test"
)

# Combine utility metrics
real_world_utility_metrics <- bind_rows(
  original_train_evaluation$metrics,
  original_test_evaluation$metrics)

print(real_world_utility_metrics)
write.csv(original_train_evaluation$predictions,
  "pred_original_train.csv", row.names = FALSE)

write.csv(original_test_evaluation$predictions,
  "pred_original_test.csv", row.names = FALSE)

write.csv(real_world_utility_metrics,
  "real_world_utility_metrics.csv", row.names = FALSE)

# ROC curve for the original test set
plot(
  original_test_evaluation$roc,
  col = "blue",
  lwd = 2,
  legacy.axes = TRUE,
  main = paste0(
    "Synthetic-Trained RF on Original Test Data\nAUC = ",
    round(original_test_evaluation$metrics$ROC_AUC, 3)))

abline(a = 0, b = 1, lty = 2, col = "grey" )

#===============================================================================
# KNN SIMILARITY: SYNTHETIC VS ORIGINAL
# Train and test sets are evaluated separately
#===============================================================================

# Stack all datasets before model.matrix so they receive identical dummy-variable columns.
n_synthetic_train <- nrow(features_synthetic)
n_original_train  <- nrow(features_original_train)
n_synthetic_test  <- nrow(features_synthetic_test)
n_original_test   <- nrow(features_original_test)

all_features <- bind_rows(
  features_synthetic,
  features_original_train,
  features_synthetic_test,
  features_original_test
)

all_matrix <- model.matrix(~ . - 1, data = all_features)

# Recover each matrix
row_start <- 1

synthetic_train_mat <- all_matrix[
  row_start:(row_start + n_synthetic_train - 1), , drop = FALSE]

row_start <- row_start + n_synthetic_train

original_train_mat <- all_matrix[
  row_start:(row_start + n_original_train - 1), , drop = FALSE]

row_start <- row_start + n_original_train

synthetic_test_mat <- all_matrix[
  row_start:(row_start + n_synthetic_test - 1), , drop = FALSE]

row_start <- row_start + n_synthetic_test

original_test_mat <- all_matrix[
  row_start:(row_start + n_original_test - 1), , drop = FALSE]

# Scale using synthetic training means and standard deviations
#-----------------------

training_means <- colMeans(synthetic_train_mat, na.rm = TRUE)

training_sd <- apply(synthetic_train_mat, 2, sd, na.rm = TRUE)

# Remove predictors with zero variance in synthetic training data
keep_knn_columns <- (!is.na(training_sd) & training_sd > 0)

synthetic_train_mat <- synthetic_train_mat[
  , keep_knn_columns, drop = FALSE]

original_train_mat <- original_train_mat[
  , keep_knn_columns, drop = FALSE]

synthetic_test_mat <- synthetic_test_mat[
  , keep_knn_columns, drop = FALSE]

original_test_mat <- original_test_mat[
  , keep_knn_columns, drop = FALSE ]

training_means <- training_means[keep_knn_columns]
training_sd <- training_sd[keep_knn_columns]

scale_using_synthetic_train <- function(x) {
  sweep( sweep(x, 2, training_means, FUN = "-"), 2, training_sd, FUN = "/")
}

synthetic_train_mat <- scale_using_synthetic_train(synthetic_train_mat)
original_train_mat <- scale_using_synthetic_train(original_train_mat)
synthetic_test_mat <- scale_using_synthetic_train(synthetic_test_mat)
original_test_mat <- scale_using_synthetic_train(original_test_mat)

# Training-set KNN similarity
#--------------------------

knn_train <- FNN::get.knnx(
  data = synthetic_train_mat,
  query = original_train_mat,
  k = 1)

knn_train_results <- data.frame(
  Original_Row = seq_len(nrow(original_train_mat)),
  Nearest_Synthetic_Row = knn_train$nn.index[, 1],
  Nearest_Neighbor_Distance = knn_train$nn.dist[, 1],
  Dataset = "Train"
)

cat("\nTraining-set KNN distance summary:\n")
print(summary(knn_train_results$Nearest_Neighbor_Distance))

cat("\nTraining-set KNN distance quantiles:\n")
print(
  quantile(
    knn_train_results$Nearest_Neighbor_Distance,
    probs = c(0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1)
  )
)

#===============================================================================
# CHECK FOR EXACT ORIGINAL–SYNTHETIC TRAINING MATCHES
#===============================================================================

# Show whether KNN returned distances that are effectively zero
distance_tolerance <- 1e-12

zero_distance_rows <- which(
  knn_train_results$Nearest_Neighbor_Distance <= distance_tolerance
)

cat("\nNumber of original training rows with KNN distance approximately zero:",
  length(zero_distance_rows), "\n")

if (length(zero_distance_rows) > 0) {
  print(knn_train_results[zero_distance_rows, ], digits = 17)
}

#-------------------------------------------------------------------------------
# Verify exact matches using ALL encoded predictors
#
# This uses all_matrix before scaling and before zero-variance columns were
# removed. Therefore, it is a stronger check than KNN distance == 0.
#-------------------------------------------------------------------------------

synthetic_train_unscaled <- all_matrix[
  seq_len(n_synthetic_train), , drop = FALSE]

original_train_unscaled <- all_matrix[
  n_synthetic_train + seq_len(n_original_train), , drop = FALSE]

# Construct a precise key from every encoded predictor
create_exact_row_key <- function(x) {
  apply(x, 1, function(row_values) {
      paste(sprintf("%.17g", row_values), collapse = "|" )
    })
}

synthetic_train_keys <- create_exact_row_key(
  synthetic_train_unscaled)

original_train_keys <- create_exact_row_key(
  original_train_unscaled)

# Find the first exact synthetic match for every original row
matched_synthetic_row <- match(original_train_keys, synthetic_train_keys)

exact_original_rows <- which(!is.na(matched_synthetic_row))

cat("\nNumber of exact original-to-synthetic training matches:",
  length(exact_original_rows),"\n")

cat("Percentage of original training rows with an exact match:",
  round(100 * length(exact_original_rows) / 
  nrow(original_train_unscaled), 4), "%\n")

# Create an exact-match report
if (length(exact_original_rows) > 0) {
  
  exact_match_results <- data.frame(
    Original_Row = exact_original_rows,
    Synthetic_Row = matched_synthetic_row[exact_original_rows],
    Original_Label = as.character(target_original_train[exact_original_rows]),
    Synthetic_Label = as.character(
        target_synthetic[matched_synthetic_row[exact_original_rows]]))
  
  exact_match_results$Label_Match <- (
    exact_match_results$Original_Label ==
      exact_match_results$Synthetic_Label)
  
  print(exact_match_results)
  write.csv(exact_match_results, "exact_train_matches.csv",
    row.names = FALSE)
  
  # Save the complete matched records for inspection
  original_matched_records <- features_original_train[
    exact_match_results$Original_Row, , drop = FALSE]
  
  synthetic_matched_records <- features_synthetic[
    exact_match_results$Synthetic_Row, , drop = FALSE]
  
  names(original_matched_records) <- paste0("Original_", 
                                      names(original_matched_records))
  
  names(synthetic_matched_records) <- paste0("Synthetic_",
                                        names(synthetic_matched_records))
  
  matched_record_comparison <- cbind(exact_match_results,
    original_matched_records, synthetic_matched_records)
  
  write.csv(matched_record_comparison,
    "exact_train_match_comparison.csv", row.names = FALSE)
  
} else {
  cat(paste("No complete exact matches were found.",
      "The zero KNN distance was caused by matching only on",
      "the retained non-zero-variance predictors.\n"))
}
min_distance <- min(knn_train$nn.dist[, 1]) 
print(min_distance, digits = 17) 
sprintf("%.17g", min_distance)
zero_matches <- which(knn_train$nn.dist[, 1] < 1e-12)
length(zero_matches)
zero_match_pairs <- data.frame(Original_Row = zero_matches, Synthetic_Row = knn_train$nn.index[zero_matches, 1], 
                                Distance = knn_train$nn.dist[zero_matches, 1] ) 

print(zero_match_pairs)
i <- zero_match_pairs$Original_Row[1] 
j <- zero_match_pairs$Synthetic_Row[1]
features_original_train[i, , drop = FALSE]
features_synthetic[j, , drop = FALSE]
original_raw <- train_original[i, rf_predictor_names, drop = FALSE]
synthetic_raw <- train_synthetic_shuffled[j, rf_predictor_names, drop = FALSE]
comparison <- data.frame(
  Variable = rf_predictor_names,
  Original = as.character(original_raw[1, ]),
  Synthetic = as.character(synthetic_raw[1, ]))

comparison$Same <- (comparison$Original == comparison$Synthetic)
print(comparison)
comparison[!comparison$Same, ]

# Test-set KNN similarity
#---------------------------
knn_test <- FNN::get.knnx(data = synthetic_test_mat,
                          query = original_test_mat, k = 1)

knn_test_results <- data.frame(
  Original_Row = seq_len(nrow(original_test_mat)),
  Nearest_Synthetic_Row = knn_test$nn.index[, 1],
  Nearest_Neighbor_Distance = knn_test$nn.dist[, 1],
  Dataset = "Test"
)

cat("\nTest-set KNN distance summary:\n")
print(summary(knn_test_results$Nearest_Neighbor_Distance))

cat("\nTest-set KNN distance quantiles:\n")
print(quantile(knn_test_results$Nearest_Neighbor_Distance,
    probs = c(0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1)))

# Save KNN results
write.csv(knn_train_results,"knn_similarity_train.csv", row.names = FALSE)
write.csv(  knn_test_results, "knn_similarity_test.csv", row.names = FALSE)

# Side-by-side KNN summary
knn_similarity_summary <- bind_rows(knn_train_results, knn_test_results
) %>%
  group_by(Dataset) %>%
  summarise(
    N = n(),
    Mean_Distance = mean(Nearest_Neighbor_Distance),
    SD_Distance = sd(Nearest_Neighbor_Distance),
    Median_Distance = median(Nearest_Neighbor_Distance),
    Q90_Distance = quantile(
      Nearest_Neighbor_Distance,
      0.90
    ),
    Q95_Distance = quantile(
      Nearest_Neighbor_Distance,
      0.95
    ),
    Maximum_Distance = max(Nearest_Neighbor_Distance),
    .groups = "drop"
  )

print(knn_similarity_summary)
write.csv(knn_similarity_summary, "knn_similarity_summary.csv", row.names = FALSE)