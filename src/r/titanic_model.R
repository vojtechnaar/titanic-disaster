# Titanic survival prediction - R / logistic regression (glm, binomial family).
#
# Every step prints what it does, so the whole analysis can be followed from
# the terminal output alone:
#
#   1. Load train.csv and explore it
#   2. Clean the data and engineer features
#   3. Fit a logistic regression and measure training accuracy
#   4. Load test.csv, apply the same changes and predict
#   5. Measure test-set accuracy and save the predictions

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

options(width = 160, dplyr.summarise.inform = FALSE)
set.seed(42)

# Inside Docker DATA_DIR=/app/data (set in the Dockerfile); locally it defaults to src/data
script_arg <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
script_dir <- if (length(script_arg) == 1) dirname(normalizePath(script_arg)) else getwd()
data_dir <- Sys.getenv("DATA_DIR", unset = normalizePath(file.path(script_dir, "..", "data"), mustWork = FALSE))
output_file <- file.path(data_dir, "predictions_r.csv")

common_titles <- c("Mr", "Mrs", "Miss", "Master")
# Fixed factor levels so train and test always share the same dummy variables.
# The first level of each is the reference level in the regression.
pclass_levels <- c("1", "2", "3")
embarked_levels <- c("C", "Q", "S")
title_levels <- c("Mr/Mrs", "Miss", "Master", "Rare")
drop_columns <- c("PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch")
model_formula <- Survived ~ Sex + Age + Fare + HasCabin + FamilySize + IsAlone + Pclass + Embarked + Title

banner <- function(title) {
  line <- strrep("=", 72)
  cat("\n", line, "\n", title, "\n", line, "\n", sep = "")
}

load_csv <- function(filename) {
  path <- file.path(data_dir, filename)
  if (!file.exists(path)) {
    cat("\nERROR: could not find ", path, "\n",
        "Download the data from https://www.kaggle.com/competitions/titanic/data,\n",
        "put train.csv and test.csv in src/data/ and mount that folder into the\n",
        "container with -v (see README.md).\n", sep = "")
    quit(status = 1)
  }
  df <- read_csv(path, show_col_types = FALSE)
  cat(sprintf("Loaded %s: %d rows x %d columns\n", filename, nrow(df), ncol(df)))
  df
}

print_rates <- function(df, group) {
  df %>%
    group_by(.data[[group]]) %>%
    summarise(survival_rate = sprintf("%.1f%%", 100 * mean(Survived))) %>%
    as.data.frame() %>%
    print(row.names = FALSE)
}

explore <- function(train) {
  banner("STEP 1 - Explore train.csv")
  cat("\nFirst 5 rows:\n")
  print(as.data.frame(head(train, 5)))
  cat("\nColumn types:\n")
  print(sapply(train, class))
  missing <- colSums(is.na(train))
  cat("\nMissing values per column:\n")
  print(missing[missing > 0])
  cat(sprintf("\nOverall survival rate: %.1f%%\n", 100 * mean(train$Survived)))
  cat("\nSurvival rate by sex:\n")
  print_rates(train, "Sex")
  cat("\nSurvival rate by passenger class:\n")
  print_rates(train, "Pclass")
}

# 'Braund, Mr. Owen Harris' -> 'Mr'. Uncommon titles are grouped as 'Rare'.
extract_title <- function(names) {
  titles <- trimws(sub("^[^,]*,\\s*([^.]*)\\..*$", "\\1", names))
  titles[titles %in% c("Mlle", "Ms")] <- "Miss"
  titles[titles == "Mme"] <- "Mrs"
  ifelse(titles %in% common_titles, titles, "Rare")
}

# Fill values come from the training data only, then are reused for test.
learn_fill_values <- function(train) {
  titles <- extract_title(train$Name)
  list(
    age_by_title = tapply(train$Age, titles, median, na.rm = TRUE),
    age_overall = median(train$Age, na.rm = TRUE),
    embarked_mode = names(which.max(table(train$Embarked))),
    fare_median = median(train$Fare, na.rm = TRUE)
  )
}

# Apply the same cleaning and feature steps to any split, printing every change.
prepare <- function(df, fill, name) {
  say <- function(fmt, ...) cat(sprintf(paste0("[%s] ", fmt, "\n"), name, ...))
  say("Starting shape: %d rows x %d columns", nrow(df), ncol(df))

  df$Title <- extract_title(df$Name)
  counts <- sort(table(df$Title), decreasing = TRUE)
  say("Added 'Title' parsed from Name (uncommon titles grouped as 'Rare'): %s",
      paste(names(counts), counts, sep = "=", collapse = ", "))

  missing_age <- is.na(df$Age)
  df$Age[missing_age] <- unname(fill$age_by_title[df$Title[missing_age]])
  df$Age[is.na(df$Age)] <- fill$age_overall
  say("Filled %d missing 'Age' values with the training median age for the passenger's title", sum(missing_age))

  n <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- fill$embarked_mode
  say("Filled %d missing 'Embarked' values with the most common port ('%s')", n, fill$embarked_mode)

  n <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fill$fare_median
  say("Filled %d missing 'Fare' values with the training median fare (%.2f)", n, fill$fare_median)

  df$HasCabin <- as.integer(!is.na(df$Cabin))
  say("Added 'HasCabin' (1 if a cabin number is recorded): %d of %d passengers", sum(df$HasCabin), nrow(df))

  df$FamilySize <- df$SibSp + df$Parch + 1
  df$IsAlone <- as.integer(df$FamilySize == 1)
  say("Added 'FamilySize' (SibSp + Parch + 1) and 'IsAlone': %d passengers travelled alone", sum(df$IsAlone))

  df$Sex <- as.integer(df$Sex == "male")
  say("Encoded 'Sex' as 0 = female, 1 = male")

  df$Title[df$Title %in% c("Mr", "Mrs")] <- "Mr/Mrs"
  say("Merged titles 'Mr' and 'Mrs' into 'Mr/Mrs': with 'Sex' in the model they add no extra information, and keeping both would make 'Sex' and 'Title' redundant")

  df <- df %>% select(-all_of(drop_columns))
  say("Dropped columns not used as features: %s", paste(drop_columns, collapse = ", "))

  df$Pclass <- factor(df$Pclass, levels = pclass_levels)
  df$Embarked <- factor(df$Embarked, levels = embarked_levels)
  df$Title <- factor(df$Title, levels = title_levels)
  say("Converted Pclass, Embarked, Title to factors (reference levels: 1, C, Mr/Mrs) -> final shape: %d rows x %d columns",
      nrow(df), ncol(df))
  df
}

predict_class <- function(model, newdata) {
  as.integer(predict(model, newdata = newdata, type = "response") > 0.5)
}

main <- function() {
  cat("Reading data from:", data_dir, "\n")

  # -------------------------------------------------------------- step 1
  train_raw <- load_csv("train.csv")
  explore(train_raw)

  # -------------------------------------------------------------- step 2
  banner("STEP 2 - Clean the data and engineer features")
  fill <- learn_fill_values(train_raw)
  cat("Median age by title (learned from train, used to fill missing ages):\n")
  print(round(fill$age_by_title, 1))
  cat("\n")
  train <- prepare(train_raw, fill, "train")
  cat("\nSurvival rate by title:\n")
  print_rates(mutate(train_raw, Title = extract_title(Name)), "Title")
  cat("\nFirst 5 rows after preparation:\n")
  print(as.data.frame(head(train, 5)))

  # -------------------------------------------------------------- step 3
  banner("STEP 3 - Train a logistic regression model")
  cat("Formula:", deparse(model_formula, width.cutoff = 200L), "\n")
  cat("Model: glm(family = binomial) - unregularized logistic regression\n")
  model <- glm(model_formula, data = train, family = binomial)

  cat("\nCoefficients (log-odds; positive = more likely to survive):\n")
  print(round(summary(model)$coefficients, 3))

  train_acc <- mean(predict_class(model, train) == train$Survived)

  folds <- sample(rep(1:5, length.out = nrow(train)))
  cv_scores <- sapply(1:5, function(k) {
    fit <- glm(model_formula, data = train[folds != k, ], family = binomial)
    mean(predict_class(fit, train[folds == k, ]) == train$Survived[folds == k])
  })
  cat(sprintf("\nTraining accuracy:               %.4f\n", train_acc))
  cat(sprintf("5-fold cross-validated accuracy: %.4f (+/- %.4f)\n", mean(cv_scores), sd(cv_scores)))

  # -------------------------------------------------------------- step 4
  banner("STEP 4 - Predict on test.csv")
  test_raw <- load_csv("test.csv")
  cat("\n")
  test <- prepare(test_raw, fill, "test")
  predictions <- predict_class(model, test)
  cat(sprintf("\nPredicted %d survivors out of %d test passengers (%.1f%%)\n",
              sum(predictions), length(predictions), 100 * mean(predictions)))

  submission <- data.frame(PassengerId = test_raw$PassengerId, Survived = predictions)
  write_csv(submission, output_file)
  cat("Saved predictions to", output_file, "\n")

  # -------------------------------------------------------------- step 5
  banner("STEP 5 - Test-set accuracy")
  test_acc <- NULL
  if ("Survived" %in% names(test_raw)) {
    test_acc <- mean(predictions == test_raw$Survived)
    label <- "true labels"
  } else {
    cat("test.csv has no 'Survived' column: Kaggle keeps the true test labels hidden.\n")
    reference_path <- file.path(data_dir, "gender_submission.csv")
    if (file.exists(reference_path)) {
      reference <- read_csv(reference_path, show_col_types = FALSE)
      merged <- inner_join(submission, reference, by = "PassengerId", suffix = c("_pred", "_ref"))
      test_acc <- mean(merged$Survived_pred == merged$Survived_ref)
      label <- "gender_submission.csv"
      cat("Using gender_submission.csv (Kaggle's sample answer: all women survive, all men die)\n")
      cat("as the reference labels for the test set.\n")
    } else {
      cat("gender_submission.csv not found either, so test accuracy cannot be computed.\n")
    }
  }
  if (!is.null(test_acc)) cat(sprintf("\nTest accuracy (vs %s): %.4f\n", label, test_acc))
  cat(sprintf("For the official score, upload %s to the Kaggle competition.\n", basename(output_file)))

  # ------------------------------------------------------------- summary
  banner("SUMMARY")
  cat(sprintf("Training accuracy:               %.4f\n", train_acc))
  cat(sprintf("5-fold cross-validated accuracy: %.4f\n", mean(cv_scores)))
  if (!is.null(test_acc)) cat(sprintf("Test accuracy (vs %s): %.4f\n", label, test_acc))
  cat(sprintf("Predictions file:                %s\n", output_file))
}

main()
