
# install.packages("caret")
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("synthpop")
# SETWD
#===============================================================================

# check the current working directory
current_directory <- getwd()

# print the current working directory
print(current_directory)

# change working directory
setwd("C:\\Experiments\\rehospitalization-r")

# check the current working directory
current_directory <- getwd()

# print the current working directory
print(current_directory)

# READ
#===============================================================================

# read the CSV file
data_train <- read.csv("train_v0.csv")
data_iid <- read.csv("test_v0.csv")
data_ood <- read.csv("ood_test_v0.csv")
data_test <- rbind(data_iid, data_ood)

# view the first few rows of the data frame
head(data_train)


# UTILS
#===============================================================================

# function to clean and standardize drug names
clean_drug_names <- function(drug_list) {
  # remove punctuation
  cleaned_drugs <- gsub("[[:punct:]]", "", drug_list)
  # convert to lowercase
  cleaned_drugs <- tolower(cleaned_drugs)
  # trim whitespace
  cleaned_drugs <- trimws(cleaned_drugs)
  return(cleaned_drugs)
}


standardize_drug_names <- function(drug_names) {
  # replace spaces and other non-alphanumeric characters with underscores
  drug_names <- gsub("[^[:alnum:]]+", "_", tolower(drug_names))
  
  # ensure names start with a letter (prepend with 'x' if not)
  drug_names <- ifelse(grepl("^[^a-zA-Z]", drug_names), paste0("x", drug_names), drug_names)
  
  # ensure the names are valid R identifiers
  drug_names <- make.names(drug_names, unique = FALSE)
  
  return(drug_names)
}

parse_list_string <- function(x) {
  # remove leading and trailing double quotes if present
  x <- gsub('^"|"$', '', x)
  # remove square brackets
  clean_string <- gsub("\\[|\\]", "", x)
  # split the string into elements based on space separation, elements are not comma-separated
  elements <- unlist(strsplit(clean_string, " '"))
  # remove single quotes from all elements
  elements <- gsub("'", "", elements)
  return(elements)
}

calculate_first_letter_counts <- function(codes) {
  first_letters <- substr(codes, 1, 1) # extract the first letter
  letter_counts <- table(first_letters) # count occurrences
  dict_str <- paste(names(letter_counts), letter_counts, sep = ": ", collapse = ", ") 
  return(dict_str)
}

# PREPROCESSING
#===============================================================================

# step 1: initialize columns
initialize_letter_columns <- function(df) {
  letters_to_include <- LETTERS # LETTERS is a constant that contains all uppercase letters
  for(letter in letters_to_include) {
    df[[letter]] <- 0
  }
  return(df)
}

# step 2: parse counts
update_letter_counts <- function(df, column_name) {
  library(stringr) # ensure stringr is available for string operations
  
  # predefine all letters of the alphabet as uppercase letters
  letters_to_include <- LETTERS
  
  # iterate over each row of the dataframe
  for(i in 1:nrow(df)) {
    # initially set default values for all letters for the current row
    for(letter in letters_to_include) {
      df[i, letter] <- 0 # set default value, e.g., 0
    }
    
    # extract matches for the current row from the specified column
    matches <- str_match_all(df[[column_name]][i], "(\\w): (\\d+)")
    
    # proceed if there are matches
    if(length(matches) > 0 && length(matches[[1]]) > 0) {
      counts <- matches[[1]]
      
      # loop through each match to update the dataframe
      for(j in 1:nrow(counts)) {
        key <- counts[j, 2] # the letter
        value <- as.numeric(counts[j, 3]) # the count, converted to numeric
        if(key %in% letters_to_include) {
          # update the specific letter's count for the current row
          df[i, key] <- value
        }
      }
    }
  }
  
  return(df)
}


# READ
#===============================================================================

suppressPackageStartupMessages(library(synthpop))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(sampling))

seed <- 20190110

# --train
data_train$Administered_Drugs <- lapply(data_train$Administered_Drugs, parse_list_string)
data_train$Administered_Drugs <- lapply(data_train$Administered_Drugs, clean_drug_names)
data_train$Administered_Drugs <- lapply(data_train$Administered_Drugs, standardize_drug_names)
unique_drugs_train <- sort(unique(unlist(data_train$Administered_Drugs))) #289

# --test
data_test$Administered_Drugs <- lapply(data_test$Administered_Drugs, parse_list_string)
data_test$Administered_Drugs <- lapply(data_test$Administered_Drugs, clean_drug_names)
data_test$Administered_Drugs <- lapply(data_test$Administered_Drugs, standardize_drug_names)
unique_drugs_test <- sort(unique(unlist(data_test$Administered_Drugs))) #251

unique_drugs = c(unique_drugs_train, unique_drugs_test) #540
# create a count column for each drug
for(drug in unique_drugs) {
  data_train[[paste(drug, "count", sep = "_")]] <- sapply(data_train$Administered_Drugs, function(x) sum(x == drug))
  data_test[[paste(drug, "count", sep = "_")]] <- sapply(data_test$Administered_Drugs, function(x) sum(x == drug))
}


# FEATURE ENGINEERING
#===============================================================================

# --train
data_train$Dx_Secondary <- lapply(data_train$Dx_Secondary, parse_list_string)
data_train$Dx_Secondary_Dict <- sapply(data_train$Dx_Secondary, calculate_first_letter_counts)

# --test
data_test$Dx_Secondary <- lapply(data_test$Dx_Secondary, parse_list_string)
data_test$Dx_Secondary_Dict <- sapply(data_test$Dx_Secondary, calculate_first_letter_counts)


# step 1: initialize the letter columns
data_train_eletters <- initialize_letter_columns(data_train)
data_test_eletters <- initialize_letter_columns(data_test)

# step 2: update df with letter counts
data_train_fletters <- update_letter_counts(data_train, "Dx_Secondary_Dict")
data_test_fletters <- update_letter_counts(data_test, "Dx_Secondary_Dict")

orig.df_train <- data_train_fletters %>% dplyr::select(-Patient, -Dx_Secondary, -Dx_Secondary_Dict, -Administered_Drugs, -Glukose, -Potassium, -Sodium, -Creatinine, -Urea, -CRP, -ERAdmissionCount, -X, -AdmissionYear, -Total_patient_visits, -PATIENT_CARDINALITY) 
orig.df_test <- data_test_fletters %>% dplyr::select(-Patient, -Dx_Secondary, -Dx_Secondary_Dict, -Administered_Drugs, -Glukose, -Potassium, -Sodium, -Creatinine, -Urea, -CRP, -ERAdmissionCount, -X, -AdmissionYear, -Total_patient_visits, -PATIENT_CARDINALITY) 

# stratification
# --train
orig.df_train <- orig.df_train %>% dplyr::mutate(AdmissionDx = substr(orig.df_train$AdmissionDx, 0, 1))
orig.df_train <- orig.df_train %>% dplyr::mutate(Dx_Discharge = substr(orig.df_train$Dx_Discharge, 0, 1))

# --test
orig.df_test <- orig.df_test %>% dplyr::mutate(AdmissionDx = substr(orig.df_test$AdmissionDx, 0, 1))
orig.df_test <- orig.df_test %>% dplyr::mutate(Dx_Discharge = substr(orig.df_test$Dx_Discharge, 0, 1))


na_count_train <- sum(is.na(orig.df_train$PreviousAdmissionDays))
print(na_count_train)

na_count_test <- sum(is.na(orig.df_test$PreviousAdmissionDays))
print(na_count_test)

orig.df_train[c("PreviousAdmissionDays", "Weight_Discharge", "Height_Discharge")] <- lapply(orig.df_train[c("PreviousAdmissionDays", "Weight_Discharge", "Height_Discharge")], function(x) ifelse(is.na(x), -8, x))
orig.df_test[c("PreviousAdmissionDays", "Weight_Discharge", "Height_Discharge")] <- lapply(orig.df_test[c("PreviousAdmissionDays", "Weight_Discharge", "Height_Discharge")], function(x) ifelse(is.na(x), -8, x))

# resolving constant values
# --train
constant_columns_train <- names(orig.df_train)[sapply(orig.df_train, function(x) length(unique(x)) == 1)]
# --test
constant_columns_test <- names(orig.df_test)[sapply(orig.df_test, function(x) length(unique(x)) == 1)]

constant_columns <- unique(c(constant_columns_train, constant_columns_test))

# remove constant columns
# --train
orig.df_train <- orig.df_train[, !names(orig.df_train) %in% constant_columns]
# --test
orig.df_test <- orig.df_test[, !names(orig.df_test) %in% constant_columns]


# DATA SYNTHESISING
#===============================================================================
# --train
synth.obj_train <- syn(orig.df_train, minnumlevels = 5, seed = seed)

# --test
synth.obj_test <- syn(orig.df_test, minnumlevels = 5, seed = seed)


# SYNCHRONISING COLUMN NAMES: TRAIN SYN = TRAIN ORIG | TEST SYN = TEST ORIG
#===============================================================================

# check if names are identical for TRAIN original and synthetic
columns_are_identical <- identical(names(orig.df_train), names(synth.obj_train$syn))
# print the result
if(columns_are_identical) {
  print("TRAIN SYN vs TRAIN ORIG have identical column names.")
} else {
  print("TRAIN SYN vs TRAIN ORIG do not have identical column names.")
}

# check if names are identical for TEST original and synthetic
columns_are_identical <- identical(names(orig.df_test), names(synth.obj_test$syn))
# print the result
if(columns_are_identical) {
  print("TEST SYN vs TEST ORIG have identical column names.")
} else {
  print("TEST SYN vs TEST ORIG do not have identical column names.")
}

# check if names are identical between original TRAIN and TEST
columns_are_identical <- identical(names(orig.df_train), names(orig.df_test))
# print the result
if(columns_are_identical) {
  print("TRAIN ORIG vs TEST ORIG have identical column names.")
} else {
  print("TRAIN ORIG vs TEST ORIG do not have identical column names.")
}

# check if names are identical between synthetic TRAIN and TEST
columns_are_identical <- identical(names(synth.obj_train$syn), names(synth.obj_test$syn))
# print the result
if(columns_are_identical) {
  print("TRAIN SYN vs TEST SYN have identical column names.")
} else {
  print("TRAIN SYN vs TEST SYN do not have identical column names.")
}


# OVERSAMPLING with SYNTHESISING MINORITY CLASS
#===============================================================================
# estimate class distribution
# --train
class_distribution <- table(orig.df_train$Label)
class_distribution
# 0       1 
# 26688   633

class_distribution_syn <- table(synth.obj_train$syn$Label)
class_distribution_syn
# 0       1 
# 26697   624 

# --test
class_distribution <- table(orig.df_test$Label)
class_distribution
# 0     1 
# 6919  139 

class_distribution_syn <- table(synth.obj_test$syn$Label)
class_distribution_syn
# 0     1 
# 6921  137 

# oversample minority class 
minority_original_train <- orig.df_train[orig.df_train$Label == "1", ]
minority_original_test <- orig.df_test[orig.df_test$Label == "1", ]

# --train
synth.obj_train_min_1 <- syn(minority_original_train, minnumlevels = 5, seed = 23)
synth.obj_train_min_2 <- syn(minority_original_train, minnumlevels = 5, seed = 32)

# --test
synth.obj_test_min_1 <- syn(minority_original_test, minnumlevels = 5, seed = 23)
synth.obj_test_min_2 <- syn(minority_original_test, minnumlevels = 5, seed = 32)

# combine
synth.obj_train_over <- rbind(synth.obj_train$syn, synth.obj_train_min_1$syn, synth.obj_train_min_2$syn)
synth.obj_test_over <- rbind(synth.obj_test$syn, synth.obj_test_min_1$syn, synth.obj_test_min_2$syn)

# check
# --train
class_distribution_syn <- table(synth.obj_train_over$Label)
class_distribution_syn
# 0      1 
# 26697  1890 
# increase: 624 -> 1890 

# --test
class_distribution_syn <- table(synth.obj_test_over$Label)
class_distribution_syn
# 0     1 
# 6921  415  
# increase: 137 -> 415


# CHECK UTILITY pMSE
#===============================================================================
# option 1 align to reference
# align_to_reference <- function(df, ref_df) {
#   
#   # align columns using reference dataframe
#   aligned_df <- df[, names(ref_df), drop = FALSE]
#   
#   # convert column types to match reference
#   for (col in names(ref_df)) {
#     
#     if (class(aligned_df[[col]]) != class(ref_df[[col]])) {
#       
#       if (is.numeric(ref_df[[col]])) {
#         aligned_df[[col]] <- as.numeric(aligned_df[[col]])
#       }
#       
#       else if (is.integer(ref_df[[col]])) {
#         aligned_df[[col]] <- as.integer(aligned_df[[col]])
#       }
#       
#       else if (is.factor(ref_df[[col]])) {
#         aligned_df[[col]] <- factor(
#           aligned_df[[col]],
#           levels = levels(ref_df[[col]])
#         )
#       }
#       
#       else if (is.character(ref_df[[col]])) {
#         aligned_df[[col]] <- as.character(aligned_df[[col]])
#       }
#       
#     }
#   }
#   
#   return(aligned_df)
# }
# option 2 align to reference
align_to_reference <- function(df, ref_df) {
  
  # 1. Align column order using reference
  aligned_df <- df[, names(ref_df), drop = FALSE]
  
  # 2. Convert character columns to factors (utility.gen prefers factors)
  aligned_df[] <- lapply(aligned_df, function(x) {
    if (is.character(x)) factor(x) else x
  })
  
  ref_df[] <- lapply(ref_df, function(x) {
    if (is.character(x)) factor(x) else x
  })
  
  # 3. Convert column types to match reference
  for (col in names(ref_df)) {
    
    if (!inherits(aligned_df[[col]], class(ref_df[[col]]))) {
      
      if (is.numeric(ref_df[[col]])) {
        aligned_df[[col]] <- as.numeric(aligned_df[[col]])
      }
      
      else if (is.integer(ref_df[[col]])) {
        aligned_df[[col]] <- as.integer(aligned_df[[col]])
      }
      
      else if (is.factor(ref_df[[col]])) {
        aligned_df[[col]] <- factor(aligned_df[[col]])
      }
      
      else if (is.character(ref_df[[col]])) {
        aligned_df[[col]] <- as.character(aligned_df[[col]])
      }
    }
    
    # 4. Synchronize factor levels
    if (is.factor(ref_df[[col]]) && is.factor(aligned_df[[col]])) {
      
      all_levels <- union(
        levels(ref_df[[col]]),
        levels(aligned_df[[col]])
      )
      
      aligned_df[[col]] <- factor(aligned_df[[col]], levels = all_levels)
      ref_df[[col]]     <- factor(ref_df[[col]], levels = all_levels)
    }
  }
  
  # 5. Ensure numeric columns are consistent (avoid integer/numeric mismatch)
  for (col in names(ref_df)) {
    if (is.numeric(ref_df[[col]])) {
      aligned_df[[col]] <- as.numeric(aligned_df[[col]])
    }
  }
  
  return(aligned_df)
}

# check for repeated samples in synthetic train data
# train
train_aligned <- align_to_reference(orig.df_train, synth.obj_train_over)
orig_strings <- apply(train_aligned, 1, paste, collapse = "_")
syn_strings  <- apply(synth.obj_train_over, 1, paste, collapse = "_")
matches <- syn_strings %in% orig_strings
sum(matches)
synth.obj_train_over[matches, ]

# test
test_aligned <- align_to_reference(orig.df_test, synth.obj_test_over)
orig_strings_test <- apply(test_aligned, 1, paste, collapse = "_")
syn_strings_test  <- apply(synth.obj_test_over, 1, paste, collapse = "_")
matches <- syn_strings_test %in% orig_strings_test
sum(matches)
synth.obj_train_over[matches, ]

# pMSE utility metric
# train
utility_result_train <- utility.gen(
  object     = synth.obj_train_over, 
  data       = train_aligned,       
  print.flag = TRUE
)
print(utility_result_train)
train_aligned
# test
utility_result_test <- utility.gen(
  object     = synth.obj_test_over, 
  data       = test_aligned,       
  print.flag = TRUE
)
print(utility_result_test)

# disclosure risk
# train
risk <- disclosure(object = synth.obj_test_over, orig = train_aligned)
print(risk)

# KNN nearest samples
library(FNN)
orig_aligned <- orig.df_train[, names(synth.obj_train_over), drop = FALSE]
syn_aligned  <- synth.obj_train_over

for (col in names(orig_aligned)) {
  
  if (is.factor(orig_aligned[[col]]) || is.character(orig_aligned[[col]])) {
    
    all_levels <- union(
      unique(orig_aligned[[col]]),
      unique(syn_aligned[[col]])
    )
    
    orig_aligned[[col]] <- factor(orig_aligned[[col]], levels = all_levels)
    syn_aligned[[col]]  <- factor(syn_aligned[[col]],  levels = all_levels)
    
  }
}
orig_mat <- model.matrix(~ . -1, data = orig_aligned)
syn_mat  <- model.matrix(~ . -1, data = syn_aligned)
nn <- get.knnx(orig_mat, syn_mat, k = 1)
summary(nn$nn.dist)

# WRITE
#===============================================================================

# unbalanced synthetic and original data sets
write.csv(synth.obj_train$syn, "train-synthetic.csv")
write.csv(orig.df_train, "train-original.csv")

write.csv(synth.obj_test$syn, "test-synthetic.csv")
write.csv(orig.df_test, "test-original.csv")

# Drop the Label generating columns
synth.obj_train_over <- synth.obj_train_over[, !(names(synth.obj_train_over) %in% c("NextAdmissionDays", "VISIT_COUNT"))]
synth.obj_test_over <- synth.obj_test_over[, !(names(synth.obj_test_over) %in% c("NextAdmissionDays", "VISIT_COUNT"))]
# synthetic data sets with oversampling
write.csv(synth.obj_train_over, "train-synthetic_over.csv")
write.csv(synth.obj_test_over, "test-synthetic_over.csv")