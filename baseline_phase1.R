
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
