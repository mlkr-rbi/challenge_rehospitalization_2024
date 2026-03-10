
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
# SETWD
#===============================================================================

# check the current working directory
current_directory <- getwd()

# print the current working directory
print(current_directory)

# change working directory
setwd("C:/Experiments/rehospitalization-r")

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
  # split the string into elements based on space separation, considering elements are not comma-separated
  elements <- unlist(strsplit(clean_string, " '"))
  # remove single quotes from all elements
  elements <- gsub("'", "", elements)
  return(elements)
}

calculate_first_letter_counts <- function(codes) {
  first_letters <- substr(codes, 1, 1) # extract the first letter
  letter_counts <- table(first_letters) # count occurrences
  dict_str <- paste(names(letter_counts), letter_counts, sep = ": ", collapse = ", ") # Create string
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

# check if names are identical
columns_are_identical <- identical(names(orig.df_train), names(orig.df_test))
# print the result
if(columns_are_identical) {
  print("The dataframes have identical column names.")
} else {
  print("The dataframes do not have identical column names.")
}



# DATA SYNTHESISING
#===============================================================================
# --train
synth.obj_train <- syn(orig.df_train, minnumlevels = 5, seed = seed)

# --test
synth.obj_test <- syn(orig.df_test, minnumlevels = 5, seed = seed)

# # Read original and synthetic training sets
# orig.df_train <- read.csv("train-original.csv")
# synth.obj_train <- list()                  # recreate the synthpop structure
# synth.obj_train$syn <- read.csv("train-synthetic.csv")
# 
# # Read original and synthetic test sets
# orig.df_test <- read.csv("test-original.csv")
# synth.obj_test <- list()
# synth.obj_test$syn <- read.csv("test-synthetic.csv")

# Original
head(orig.df_train$Discharge_Specialty, 20)

# Synthetic
head(synth.obj_train$syn$Discharge_Specialty, 20)
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
table(orig.df_train$Discharge_Specialty, useNA = "ifany")
table(synth.obj_train$syn$Discharge_Specialty, useNA = "ifany")


library(scales)

# VISUALISATION 
#===============================================================================
library(ggplot2)
# explanation-worthy variables from teams
# Explanation-worthy variables
safe_number_label <- function() {
  function(x) {
    out <- tryCatch(
      scales::label_number(scale_cut = scales::cut_short_scale())(x),
      error = function(e) scales::label_number()(x)
    )
    out
  }
}
#library(extrafont)
#font_import()          # scans installed fonts
#loadfonts(device="win") 

explanation_worthy <- c("PreviousAdmissionDays", "Dx_Discharge", 
                        "AdmissionDx", "LOS", "Discharge_Status", 
                        "Discharge_Specialty")
                                                                        
clean_cat <- function(x) {
  x <- as.character(x)
  x <- str_replace_all(x, "\u00A0", " ")  # non-breaking space -> space
  x <- str_squish(x)                      # collapse internal runs of spaces
  x <- trimws(x)                          # trim leading/trailing spaces
  x[x == ""] <- NA                        # turn empty strings into NA
  x
} 

# Freedman–Diaconis bin width
fd_binwidth <- function(x) {
  n <- length(x)
  if (n < 2) return(NA)
  bw <- 2 * IQR(x) / (n^(1/3))
}
# HISTOGRAMS PLOT OF INDIVIDUAL CATEGORICAL AND NUMERICAL VARIABLES 

plot_variable <- function(varname, 
                          orig.df, 
                          synth.df, 
                          cols = c("#01b4dd", "#24195c")) {
  cols <- c("#01b4dd", "#24195c")
  
  # Function to plot variable
  orig_var <- orig.df[[varname]]
  syn_var  <- synth.df[[varname]]
  
  # Try converting to numeric
  orig_num <- suppressWarnings(as.numeric(as.character(orig_var)))
  syn_num  <- suppressWarnings(as.numeric(as.character(syn_var)))
  
  is_numeric <- !all(is.na(orig_num)) & !all(is.na(syn_num)) # at least one numeric value in both
  
  if (is_numeric & varname != "Discharge_Specialty") {
    # Remove NA / non-finite
    orig_num <- orig_num[is.finite(orig_num)]
    syn_num  <- syn_num[is.finite(syn_num)]
    
    plot_data <- data.frame(
      Value = c(orig_num, syn_num),
      Type = rep(c("Original", "Synthetic"), times = c(length(orig_num), 
                                                       length(syn_num)))
    )
    
    if (varname == "PreviousAdmissionDays"){
      bw <- 500
      p <- ggplot(plot_data, aes(x = Value, fill = Type)) +
        geom_histogram(alpha = 0.95, binwidth = bw, width = 0.7,
                       position = position_dodge2(preserve = "single", width = 0.8)) +

        scale_y_continuous(trans ="log1p",
                           breaks = c(1, 10, 100, 1000, 10000),
                           labels = scales::comma) +
        # scale_y_log10() +  
        
        scale_fill_manual(values = cols) +
        theme_minimal() +
        theme(
          panel.grid.major.x = element_line(color = "grey50", size = 0.5, linetype = "dashed"),
          panel.grid.major.y = element_line(color = "grey50", size = 0.5, linetype = "dashed"),
          text = element_text(),     # all text in IBM Plex San
          axis.text = element_text(size = 14, color = "black"),
          legend.text = element_text(size = 14),
          legend.title = element_blank(),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, size = 0.25),
          plot.title = element_text(hjust = 0.5, size = 16)
        ) +
        labs(title = varname, x = "Value", y = "Frequency")
    } else {
      bw <- fd_binwidth(plot_data$Value)
      print(bw)
      p <- ggplot(plot_data, aes(x = Value, fill = Type)) +
        geom_histogram(alpha = 0.95, binwidth = bw, width = 0.8,
                       position = position_dodge2(preserve = "single", width = 0.8)) +
        scale_x_log10(labels = comma) +    # log10 x-axis with nice labels +
        scale_y_continuous(trans ="log1p",
                           breaks = c(1, 10, 100, 1000, 10000),
                           labels = scales::comma) +
        # scale_y_log10() +  
        
        scale_fill_manual(values = cols) +
        theme_minimal() +
        theme(
          panel.grid.major.x = element_line(color = "grey50", size = 0.5, linetype = "dashed"),
          panel.grid.major.y = element_line(color = "grey50", size = 0.5, linetype = "dashed"),
          text = element_text(),     # all text in IBM Plex San
          axis.text = element_text(size = 14, color = "black"),
          legend.text = element_text(size = 14),
          legend.title = element_blank(),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, size = 0.25),
          plot.title = element_text(hjust = 0.5, size = 16)
        ) +
        labs(title = varname, x = "Value", y = "Frequency")
    }
    
  } else {
    # Categorical
    orig_var <- clean_cat(orig_var)
    syn_var  <- clean_cat(syn_var) 
    orig_var <- as.character(orig_var)
    syn_var  <- as.character(syn_var)
    types <- c("Original", "Synthetic") 

    all_levels <- sort(unique(c(orig_var, syn_var)))
    # all_levels <- all_levels[all_levels != ""] 
    print(all_levels)
    if (varname == "Discharge_Status") {
      # Create mapping codes: D1, D2, ...
      codes <- paste0("D", seq_along(all_levels))
      code_mapping <- setNames(codes, all_levels)   # long -> code
      legend_mapping <- setNames(all_levels, codes) # code -> long
      
      # Replace with codes
      orig_codes <- code_mapping[orig_var]
      syn_codes  <- code_mapping[syn_var]
      
      plot_data <- data.frame(
        Category = factor(c(orig_codes, syn_codes), levels = codes),
        Type = rep(c("Original", "Synthetic"), 
                   times = c(length(orig_codes), length(syn_codes)))
      )
      
      # Add caption with mapping
      mapping_caption <- paste(paste(names(legend_mapping), 
                        legend_mapping, sep=": "), collapse="\n")
      # mapping_caption_wrapped <- str_wrap(mapping_caption, width = 80)
      
      p <- ggplot(plot_data, aes(x = Category, fill = Type)) +
        geom_bar(alpha = 0.95, stat = "count", 
                 position = position_dodge2(width = 0.8))+
      # , position = 'dodge', width = 0.8) +
        scale_fill_manual(values = cols) +
        theme_minimal() +
        scale_y_log10(breaks = c(1, 10, 100, 1000, 10000),
                      labels = scales::comma) +    
        theme(
          panel.grid.major.y = element_line(color = "grey50", size = 0.5, linetype = "dashed"),
          text = element_text(family = "Arial"),
          axis.text = element_text(size = 14, color = "black"),
          legend.text = element_text(size = 14),
          legend.title = element_blank(),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", 
                                      fill = NA, size = 0.25),
          plot.title = element_text(hjust = 0.5, size = 16)
        ) +
        labs(title = varname, 
             x = "Category", y = "Count")
      
      mapping_plot <- ggplot() +
        annotate("text", x = 0, y = 0, hjust = 0, vjust = 0,
                 label = mapping_caption, size = 6) +
        xlim(0, 1) + ylim(0, 1) +
        theme_void() +
        theme(
          plot.margin = margin(10, 10, 10, 10)
        )
      print(mapping_plot)
      
    } else if (varname == "Discharge_Specialty") {
      # Create mapping codes: D1, D2, ...
      # Numeric codes
      all_levels <- c(
        "3010100", 
        "3100400", 
        "3100600", 
        "3190100", 
        "3190200"
      )
      
      # Corresponding long names
      all_labels <- c(
        "Kardiologija",
        "Kardijalna kirurgija",
        "Vaskularna kirurgija",
        "Anesteziologija - Interna medicina",
        "Anesteziologija - Kirurgija"
      )
      # Short codes for x-axis
      codes <- paste0("S", seq_along(all_levels))
      
      # Mappings
      code_mapping <- setNames(codes, all_levels)     # numeric code -> S1,S2,...
      legend_mapping <- setNames(all_labels, codes)   # S1,S2,... -> full name
      
      # Replace with codes
      orig_codes <- code_mapping[orig_var]
      syn_codes  <- code_mapping[syn_var]
      
      plot_data <- data.frame(
        Category = factor(c(orig_codes, syn_codes), levels = codes),
        Type = rep(c("Original", "Synthetic"), 
                   times = c(length(orig_codes), length(syn_codes)))
      )
      
      # Add caption with mapping
      mapping_caption <- paste(paste(names(legend_mapping), 
                                     legend_mapping, sep=": "), collapse="\n")
      # mapping_caption_wrapped <- str_wrap(mapping_caption, width = 80)
      
      p <- ggplot(plot_data, aes(x = Category, fill = Type)) +
        geom_bar(alpha = 0.95, stat = "count", 
                 position = position_dodge2(width = 0.8))+
        # , position = 'dodge', width = 0.8) +
        scale_fill_manual(values = cols) +
        theme_minimal() +
        scale_y_log10(breaks = c(1, 10, 100, 1000, 10000),
                      labels = scales::comma) +    
        theme(
          panel.grid.major.y = element_line(color = "grey50", size = 0.5, linetype = "dashed"),
          text = element_text(family = "Arial"),
          axis.text = element_text(size = 14, color = "black"),
          legend.text = element_text(size = 14),
          legend.title = element_blank(),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", 
                                      fill = NA, size = 0.25),
          plot.title = element_text(hjust = 0.5, size = 16)
        ) +
        labs(title = varname, 
             x = "Category", y = "Count")
      
      mapping_plot <- ggplot() +
        annotate("text", x = 0, y = 0, hjust = 0, vjust = 0,
                 label = mapping_caption, size = 6) +
        xlim(0, 1) + ylim(0, 1) +
        theme_void() +
        theme(
          plot.margin = margin(10, 10, 10, 10)
        )
      print(mapping_plot)
    }
    else {
      # Default case for other categoricals¸
      label_mapping <- setNames(seq_along(all_levels), all_levels)
      # Filter out "" in both data frames

      orig.df <- orig.df %>%
        filter(.data[[varname]] != "") %>%
        mutate(!!varname := label_mapping[as.character(.data[[varname]])])
      
      synth.df <- synth.df %>%
        filter(.data[[varname]] != "") %>%
        mutate(!!varname := label_mapping[as.character(.data[[varname]])])
      
      plot_data <- data.frame(
        Category = factor(c(orig.df[[varname]], synth.df[[varname]]), 
                          levels = seq_along(all_levels)),
        Type = rep(c("Original", "Synthetic"), times = c(nrow(orig.df), nrow(synth.df)))
      )
      count_df <- tibble(
        Category = factor(c(orig_var, syn_var), levels = all_levels),
        Type     = factor(c(rep(types[1], length(orig_var)),
                            rep(types[2], length(syn_var))), levels = types)
      ) %>%
        tidyr::drop_na(Category) %>%
        count(Category, Type, .drop = FALSE) %>%
        complete(Category, Type, fill = list(n = 0)) %>%
        mutate(n_plot = ifelse(n == 0, NA, n))  # NA so log10() doesn't choke
      
      p <- ggplot(count_df, aes(Category, n_plot, fill = Type)) +
        geom_col(position = position_dodge2(preserve = "single", width = 0.8),
                 alpha = 0.95, width = 0.8) +
        scale_fill_manual(values = cols, breaks = c("Original","Synthetic")) +  # <- here
        scale_x_discrete(drop = FALSE) +
        scale_y_log10(breaks = c(1, 10, 100, 1000, 10000),
                      labels = scales::comma) +   
        theme(
          panel.grid.major.y = element_line(color = "grey50", size = 0.5, linetype = "dashed"),
          panel.background = element_rect(fill = "white", color = NA),
          plot.background  = element_rect(fill = "white", color = NA),
          text = element_text(family = "Arial"),
          axis.text = element_text(size = 14, color = "black"),
          legend.text = element_text(size = 14),
          legend.title = element_blank(),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, size = 0.25),
          plot.title = element_text(hjust = 0.5, size = 16)
        ) +
        labs(title = varname, x = "Category", y = "Count")
      
    }
  }
    
  return(p)
}

p <- plot_variable("LOS", orig.df_train, synth.obj_train$syn)
print(p)
# Loop over all explanation-worthy variables
plots <- lapply(explanation_worthy, function(v) {
  plot_variable(v, orig.df_train, synth.obj_train$syn)
})

for (p in plots) {
  print(p)
}

# HEATMAP VISUALIZZATION FOR EXPLANATION WORTHY COLUMNS
#===============================================================================
variables_to_include <- c("PreviousAdmissionDays", "Dx_Discharge", "AdmissionDx", "LOS", "Discharge_Status", "Discharge_Specialty")
utility.tables(synth.obj_train, orig.df_train, vars = variables_to_include, tables = "twoway", nworst = 4, print.tabs = TRUE)


u <- utility.tables(
  synth.obj_train, orig.df_train,
  vars       = c("PreviousAdmissionDays","Dx_Discharge","AdmissionDx",
                 "LOS","Discharge_Status","Discharge_Specialty"),
  tables     = "twoway",
  plot.stat  = "S_pMSE",      # choose what to colour by
  low        = "grey98",      # low colour
  high       = "#d83187",     # high colour
  max.scale  = 3,             # fix the top of the colour scale
  min.scale  = 0,             # and the bottom
  plot.title = "Two-way utility (S_pMSE)",
  n.breaks   = 6              # or use 'breaks = c(0,0.5,1,2,3)'
)

print(u)

p <- u$utility.plot +
  scale_fill_steps(                 # discrete steps for a continuous var
    low       = "grey98",
    high      = "#d83187",
    limits    = c(0, 3),
    n.breaks  = 6,
    guide     = guide_colorsteps() # step legend, not a smooth bar
  )

ggsave("utility_matrix.pdf", p, width = 8, height = 6) 

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


# WRITE
#===============================================================================

# unbalanced synthetic and original datasets
write.csv(synth.obj_train$syn, "train-synthetic.csv")
write.csv(orig.df_train, "train-original.csv")

write.csv(synth.obj_test$syn, "test-synthetic.csv")
write.csv(orig.df_test, "test-original.csv")

# synthetic datasets with oversampling
write.csv(synth.obj_train_over, "train-synthetic_over.csv")
write.csv(synth.obj_test_over, "test-synthetic_over.csv")