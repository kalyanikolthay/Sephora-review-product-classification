# ============================================
# Predicting Product Success in Beauty Retail Using Customer Reviews
# MSc Business Analytics dissertation, Queen's University Belfast (2026)
# Author: Kalyani Kolthay
#
# How to run:
# 1. Run this script from the root of the repository
# 2. Download the Sephora Products and Skincare Reviews dataset from Kaggle
#    and place product_info.csv and the reviews_*.csv files in data/
# 3. The zero-shot extraction needs a Python environment with the
#    transformers package installed. Set RETICULATE_PYTHON to the path of
#    that Python executable before running (for example in .Renviron)
# ============================================

# ============================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================
#Creating project folders (run this script from the repository root)
dir.create("data", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)

#Loading relevant libraries 
library(caret)
library(corrplot)
library(diptest)
library(flextable)
library(fs)
library(glmnet)
library(irr)
library(kernlab)
library(kknn)
library(MLmetrics)
library(openxlsx)
library(patchwork)
library(pROC)
library(randomForest)
library(reticulate)
library(scales)
library(SHAPforxgboost)
library(smotefamily)
library(tidytext)
library(tidyverse)
library(topicmodels)
library(themis)
library(vroom)
library(xgboost)

#Python environment for zero-shot extraction (set RETICULATE_PYTHON to your Python path)
if (Sys.getenv("RETICULATE_PYTHON") != "") {
  use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
}
transformers <- import("transformers")
classifier <- transformers$pipeline(
  "zero-shot-classification",
  model = "facebook/bart-large-mnli"
)
# ============================================
# SECTION 2: VISUAL LANGUAGE SETUP
# ============================================
sephora_colors <- c(
  blush = "#F4C2C2",
  mauve = "#C17B7B",
  rose = "#E8A0A0",
  cream = "#FDF0E8",
  dusty_pink = "#D4A5A5",
  soft_plum = "#9B6B7B",
  champagne = "#F7E7CE",
  warm_grey = "#B5A5A5") #Stylised using AI
theme_sephora <- function (){
  theme_minimal()+
    theme(
      plot.title = element_text(face = "bold", size = 14, color = "#9B6B7B"),
      plot.subtitle = element_text(color = "#B5A5A5", size = 11),
      axis.title = element_text(color = "#C17B7B", size = 11),
      axis.text = element_text(color = "#9B6B7B"),
      panel.grid.major = element_line(color = "#FDF0E8"),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "#FFFFFF", color = NA),
      legend.position = "bottom"
    )
}

# ============================================
# SECTION 3: DATA LOADING
# ============================================
products <- read_csv("data/product_info.csv")
review_files <- dir_ls("data/",
                       regexp = "reviews_.*\\.csv$")
reviews <- vroom(review_files,
                 col_types = cols(
                   author_id = col_character ()))

# ============================================
# SECTION 4: DATA CLEANING
# ============================================

#Product level data cleaning 
summary(products) #278 products wiht no reviews at all 

products$brand_id <- as.character(products$brand_id)

products <- products %>% 
  mutate(across(c(
    new,
    online_only,
    out_of_stock,
    limited_edition,
    sephora_exclusive,
    highlights,
    primary_category,
    secondary_category,
    tertiary_category), as.factor),
    price_tier = case_when(
      price_usd <20 ~ "Budget",
      price_usd <50 ~ "Mid Range",
      price_usd <100 ~ "Premium",
      TRUE ~ "Luxury") %>% 
      factor(levels = c("Budget", "Mid Range", "Premium", "Luxury"))) %>% 
  select(-any_of(c("value_price_usd", "child_count", "child_max_price", "child_min_price"))) 
summary(products)

#Creating a missingness helper function 
missingness_table <- function(data, caption){
  data %>% 
    summarise(across(everything(), ~ sum(is.na(.)))) %>% 
    pivot_longer(everything(),
                 names_to = "Column", values_to = "missing_count") %>% 
    mutate(missing_percent = round(missing_count/ nrow(data)*100,2)) %>% 
    arrange(desc(missing_percent)) %>% 
    flextable() %>% 
    set_header_labels(Column = "Column",
                      missing_count = "Missing Count",
                      missing_percent = "Missing (%)") %>% 
    theme_vanilla()%>% 
    autofit() %>% 
    bold(part = "header") %>% 
    color(~ missing_percent> 50, ~missing_percent, color = "red") %>% 
    color(~ missing_percent> 10 & missing_percent <=50, ~missing_percent, color = "orange") %>% 
    color(~missing_percent>0  & missing_percent <=10, ~missing_percent, color = "green") %>% 
    color(~missing_percent == 0, ~missing_percent, color = "darkgreen") %>% 
    set_caption(caption)
}

product_missingness <- missingness_table(products, 
                                         "Missingness by Column: Product Level")

product_missingness

#Removing columns with significant missingness 
products <- products %>% 
  select(-any_of(c("sale_price_usd", "variation_desc", "variation_value", "variation_type", "tertiary_category"))) #Tertiary category was removed because its redundant with secondary category and adds a lot of cardinality
cleaned_product_missingness <- missingness_table(products, "Missingness by Column: Product Level After Cleaning")
cleaned_product_missingness


#Review level data cleaning 
summary(reviews)

reviews <- reviews %>% 
  mutate(is_recommended = as.factor(is_recommended),
         review_rating = rating,
         submission_time = as.Date(submission_time))%>% 
  select(-any_of(c(
    "rating",
    "skin_tone",
    "eye_color",
    "skin_type",
    "hair_color",
    "...1",
    "product_name",
    "brand_name",
    "price_usd"
  )))
summary(reviews)
review_missingness <- missingness_table( reviews, "Missingness by Column: Review Level Missingness")
review_missingness

#Cleaning verification 
nrow(products)
nrow(reviews)

# ============================================
# SECTION 5: DEFINING POPULATIONS
# ============================================

#Baseline population: All products 
baseline_products <- products

#Analysis population 
reviewed_ids <- reviews %>%  distinct(product_id) %>%  pull(product_id)
analysis_products <- products %>% 
  filter(product_id %in% reviewed_ids)

cat("Baseline population:", nrow(baseline_products), "products\n")
cat("Analysis population:", nrow(analysis_products), "products\n")
analysis_products %>% count(primary_category)

# ============================================
# SECTION 6: REVIEW TO PRODUCT AGGREGATION
# ============================================

review_features <- reviews %>% 
  group_by(product_id) %>% 
  summarise(
    percent_recommended = mean(as.numeric(as.character(is_recommended)), na.rm =TRUE),
    mean_helpfulness = mean(helpfulness, na.rm = TRUE),
    total_feedback = sum(total_feedback_count, na.rm = TRUE),
    total_pos_feedback_count = sum(total_pos_feedback_count, na.rm = TRUE),
    total_neg_feedback_count = sum(total_neg_feedback_count, na.rm = TRUE),
    n_reviews_text = n(),
    days_active = as.numeric(max(submission_time, na.rm = TRUE)- min(submission_time, na.rm =TRUE)),
    review_velocity = round(n_reviews_text/ (days_active + 1),4),
    .groups = "drop"
  ) %>% 
  mutate(across(where(is.numeric), ~ifelse(is.nan(.), NA,.)))

analysis_products <- analysis_products %>% 
  inner_join(review_features, by = "product_id")

cat("Analysis rows:", nrow(analysis_products), "\n")
cat("New columns present:", all(c("percent_recommended", "mean_helpfulness", "review_velocity") %in% names(analysis_products)), "\n")

missingness_table(analysis_products, "Missingness: Analysis Population After Review Feaures Inclusion")

# ============================================
# SECTION 7: EDA: CORRELATION ANALYSIS
# ============================================
num_features <-analysis_products %>% 
  select(loves_count, rating, price_usd, reviews, percent_recommended, total_pos_feedback_count, total_neg_feedback_count, days_active, review_velocity) %>% 
  mutate(across(everything(), ~replace_na(., median(.,na.rm = TRUE))))

cor_mat <- cor(num_features, use = "complete.obs")
corrplot(cor_mat,
         method = "color",
         col= colorRampPalette(c("#F4C2C2", "#FFFFFF", "#9B6B7B"))(200),
         type = "upper",
         tl.col = "#9B6B7B", tl.cex = 0.8,
         addCoef.col = "#333333", number.cex = 0.7,
         title = "EDA: Structured Feature Correlations (skincare)",
         mar = c(0, 0, 2, 0)) #%recommended is highly correlated with rating. shown here to evidence the r=0.90 with rating that motivates its exclusion in Section 10

# ============================================
# SECTION 8: TARGET DIAGNOSTICS
# ============================================
#Full catalogue target diagnostics 
target_diag <- analysis_products %>% 
  filter(!is.na(rating)) %>% 
  mutate(
    z_rating = as.numeric(scale(rating)),
    z_loves = as.numeric(scale(log10(loves_count+1))),
    score_equal = 0.5*z_rating + 0.5*z_loves)

#Weighting for loves vs rating
pca <- prcomp(target_diag %>% select(z_rating,z_loves), scale.=TRUE)
print(pca$rotation)
print(summary(pca)) #Both explain nearly half of the variance therefore a 50/50 weightage justified
cor(target_diag$z_rating, target_diag$z_loves) #Validated by minimal correlation

#Threshold selection for binary target 
dip_result <- dip.test(target_diag$score_equal)
print(dip_result) #Unimodal data 

ggplot(target_diag, aes(x = score_equal))+
  geom_histogram(aes(y=after_stat(density)), bins = 40,
                 fill = "#C17B7B", color = "#FDF0E8")+
  geom_vline(aes(xintercept = quantile(score_equal, 0.75)),
             linetype = "dashed", color = "#9B6B7B", linewidth = 0.6) +
  annotate("text", x = quantile(target_diag$score_equal, 0.75),
           y = Inf, label = "Top-quartile cut", vjust = 2, hjust = -0.05,
           color = "#9B6B7B", size = 3.2) +
  labs(title = "Structure of Success Score",
       x= "Success score (equal weighting)", y = "Density")+ 
  theme_sephora()

# ============================================
# SECTION 9: TARGET CONSTRUCTION
# ============================================
model_data <- analysis_products %>% 
  filter(!is.na(rating))

provisional <- 0.5*as.numeric(scale(model_data$rating))+
  0.5*as.numeric(scale(log10(model_data$loves_count+1)))
model_data$strata_bucket<- cut(provisional,
                               breaks= quantile(provisional, probs= seq(0,1,0.25)),
                               include.lowest = TRUE, labels = FALSE) %>% as.factor()
set.seed(40485968)
train_index <- createDataPartition(model_data$strata_bucket, p =0.8, list = FALSE)
train_raw <- model_data[train_index,]
test_raw <- model_data[-train_index,]

rating_mu <- mean(train_raw$rating)
rating_sd <- sd(train_raw$rating)
loves_mu <- mean(log10(train_raw$loves_count +1))
loves_sd <- sd(log10(train_raw$loves_count+1))

score_fn <- function(df){
  zr <- (df$rating - rating_mu)/ rating_sd
  zl <-(log10(df$loves_count+1)- loves_mu)/loves_sd
  0.5*zr+0.5*zl
}
threshold <- quantile(score_fn(train_raw),0.75)

label_fn <- function(df){
  factor(ifelse(score_fn(df)>= threshold, "High", "Low"), levels = c("Low", "High"))
}
train_raw$success_label <- label_fn(train_raw)
test_raw$success_label <- label_fn(test_raw)

cat("Train balance:\n"); print(prop.table(table(train_raw$success_label)))
cat("Test balance:\n");  print(prop.table(table(test_raw$success_label)))

#Sensitivity analysis 

weight_sensitivity <- function(w, train_raw, test_raw){
  rating_mu <- mean (train_raw$rating)
  rating_sd <- sd(train_raw$rating)
  loves_mu <- mean(log10(train_raw$loves_count +1))
  loves_sd <- sd(log10(train_raw$loves_count +1))
  
  score_fn_w <- function(df){
    zr <- (df$rating - rating_mu)/ rating_sd
    zl <- (log10(df$loves_count +1)- loves_mu)/ loves_sd
    w * zr + (1-w) *zl
  }
  
  threshold_w <- quantile(score_fn_w(train_raw),0.75)
  factor(ifelse (score_fn_w(bind_rows(train_raw, test_raw))>= threshold_w, "High", "Low"),
         levels = c("Low", "High"))
}
original_labels <- weight_sensitivity(0.5, train_raw, test_raw)

weights_to_test <- c(0.3,0.4,0.5,0.6,0.7)

sensitivity_results <- map_dfr(weights_to_test, function(w){
  alt_labels <- weight_sensitivity(w, train_raw, test_raw)
  agreement <- mean(alt_labels == original_labels)
  kappa_val <- irr::kappa2(data.frame(origincal= original_labels, alt = as.character(alt_labels)))$value
  tibble(weight_rating = w,
         weight_loves = round(1-w,1),
         pct_agreement = round(agreement*100,1), 
         kappa= round(kappa_val,3))
})
sensitivity_results

# ============================================
# SECTION 10: FEATURE FIREWALL
# ============================================

firewall <- c("rating", "loves_count", "percent_recommended", "product_id", "product_name", "brand_id", "strata_bucket")
train <- train_raw %>% select(-any_of(firewall))
test <- test_raw %>% select(-any_of(firewall))

cat("Constituents leaked:",
    any(c("rating","loves_count") %in% names(train)), "\n")   # expect FALSE
cat("Target present:",
    "success_label" %in% names(train), "\n") 

# ============================================
# SECTION 11: VISUAL EDA
# ============================================

#Figure 1: Category compostion 
cat_composition <- baseline_products %>% 
  filter(!is.na(primary_category)) %>% 
  count(primary_category) %>% 
  mutate(primary_category = fct_reorder(primary_category,n))

ggplot(cat_composition, aes(x= primary_category, y = n, fill = primary_category ))+
  geom_col(width = 0.7)+
  geom_text(aes(label = scales::comma(n)), hjust= -0.15, size = 3, color = "black")+
  coord_flip() +
  scale_y_continuous(expand= expansion (mult= c(0,0.15)))+
  scale_fill_manual(values=  colorRampPalette(c("#E8CFC9", "#9B6B7B"))(n_distinct(cat_composition$primary_category)))+
  labs(title = "Category Composition of the Sephora Catalogue", x= NULL, y = "Number of products")+
  theme_sephora()+ 
  theme(legend.position = "none")

#Figure 2: price Analysis 
ggplot(baseline_products%>% filter(!is.na(primary_category), price_usd>0),
       aes(x= fct_reorder(primary_category, price_usd, median), y= price_usd, fill = primary_category))+
  geom_boxplot(alpha = 0.75, outlier.size=0.4, outlier.alpha =0.2)+
  coord_flip()+
  scale_y_log10(labels = scales::dollar)+
  scale_fill_manual(values = colorRampPalette(c("#E8CFC9", "#9B6B7B"))(n_distinct(baseline_products$primary_category)))+
  labs(title= "Price Distribution Across Product Categories", x= NULL, y= "Price (USD, log scale)")+
  theme_sephora()+
  theme(legend.position = "none")

# Figure 3: Target analysis 
target_frame <- bind_rows (
  train_raw %>% mutate(partition= "train"),
  test_raw %>% mutate(partition= "test") 
) %>% 
  select(product_id, rating, loves_count, success_label, partition) %>% 
  filter(!is.na(success_label), !is.na(rating), loves_count > 0)

ggplot(target_frame, aes(x= rating, y = loves_count, color = success_label))+
  geom_point(alpha= 0.5, size = 1.6)+
  scale_y_log10(labels = scales::comma)+
  scale_color_manual(values = c("Low" = "#B5A5A5", "High"= "#9B6B7B"), name = "Product class")+
  labs(
    title = "Product Success in Rating and Popularity Space", 
    x= "Average rating", y = "Loves Count (log scale)")+
  theme_sephora()+
  theme(legend.position = "bottom")

#Figure 4: High Product Success Rate Across Price Tiers
tier_composition <- target_frame %>% 
  left_join(analysis_products %>%  select(product_id, price_tier), by= "product_id") %>% 
  filter(!is.na(price_tier)) %>% 
  group_by(price_tier) %>% 
  summarise(
    n= n(),
    n_high = sum(success_label == "High"),
    pct_high = n_high/n,
    .groups = "drop"
  ) %>% 
  mutate(price_tier = factor(price_tier, levels = c("Budget", "Mid Range", "Premium", "Luxury")))

ggplot(tier_composition, aes(x= price_tier, y = pct_high, fill = price_tier))+
  geom_col (width=0.7)+
  geom_text(aes(label = scales:: percent(pct_high, accuracy = 0.1)),
            vjust = -0.4, size= 3.3, color = "black")+
  geom_hline(yintercept = mean(target_frame$success_label == "High"), linetype = "dashed", color = "#9B6B7B", linewidth = 0.6)+
  annotate("text", x= 0.7, y = mean(target_frame$success_label== "High")+0.02,
           label = "Overall High Rate", hjust=0, size = 2.9, color = "#9B6B7B")+
  scale_fill_manual(values = c("#E8CFC9", "#D4A5A5", "#C17B7B", "#9B6B7B"))+
  scale_y_continuous(labels = scales::percent, expand = expansion(mult= c(0,0.12)))+
  labs(
    title= "High Product Success Concentrated in Budget Price Tiers", 
    x= NULL, y = "% classified High-Success"
  )+
  theme_sephora()+
  theme(legend.position = "none")
# ============================================
# SECTION 12: FEATURE PREP FOR MODELLING
# ============================================

drop_cols <- c("brand_name", "highlights", "ingredients", "primary_category", "size", "mean_helpfulness",  "n_reviews_text", "total_pos_feedback_count", "total_neg_feedback_count")
train_final <- train %>% select(-any_of(drop_cols))
test_final <- test %>% select(-any_of(drop_cols))

#Separating targets 
y_train <- train_final$success_label
y_test <- test_final$success_label

#Building predictor frames
train_x <- train %>% 
  select(-any_of(c(drop_cols, "success_label"))) %>% 
  mutate(across(where(is.factor),droplevels))

test_x <- test %>% 
  select(-any_of(c(drop_cols, "success_label"))) %>% 
  mutate(across(where(is.factor), droplevels))

#Identifying rare secondary_category levels
rare_threshold <- 75

rare_levels <- train_x %>% 
  count(secondary_category ) %>% 
  filter(n< rare_threshold) %>% 
  pull(secondary_category) %>% 
  as.character()
cat("Collapsing", length(rare_levels), "rare subcategories into 'Other':\n")
print(rare_levels)

collapse_rare <- function(df,rare){
  df %>% 
    mutate(secondary_category = fct_other(secondary_category, drop = rare, other_level = "Other") %>% 
             droplevels())
}

train_x <- collapse_rare(train_x, rare_levels)
test_x  <- collapse_rare(test_x, rare_levels)

test_x$secondary_category <- factor(test_x$secondary_category,
                                    levels = levels(train_x$secondary_category))

#Dummy encoding factors 
dummy_fun <- dummyVars("~.", data = train_x, fullRank = TRUE)
train_matrix <- as.data.frame( predict(dummy_fun, train_x))
test_matrix <- as.data.frame(predict(dummy_fun, test_x))

cat("Train_matrix:", dim(train_matrix)[1], "x", dim(train_matrix)[2], "\n")
cat("Test_matrix:", dim(test_matrix)[1], "x", dim(test_matrix)[2], "\n")
cat("Columns match:", identical(names(train_matrix), names(test_matrix)), "\n")
cat("Targets:", length(y_train), "train", length(y_test), "test\n")
# ============================================
# SECTION 13: CORRELATION ANALYSIS FOR MODELLING
# ============================================

model_num <- train_matrix %>% 
  select(price_usd, reviews, total_feedback,days_active, review_velocity)
cor_model <- cor(model_num, use= "complete.obs")
corrplot(cor_model,
         method= "color",
         col = colorRampPalette(c("#F4C2C2", "#FFFFFF", "#9B6B7B"))(200),
         type = "upper",
         tl.col = "#9B6B7B", tl.cex = 0.8,
         addCoef.col = "#333333", number.cex = 0.7,
         title = "Modelling Feature Correlations (train)",
         mar = c(0, 0, 2, 0))

high_cor <- which(abs(cor_model)>0.8 & abs(cor_model)<1, arr.ind = TRUE)
if(nrow(high_cor)>0){
  cat("Pairs with |r| > 0.8 remaining:\n")
  for(i in seq_len(nrow(high_cor))){
    if (high_cor[i,1]< high_cor[i,2]){
      cat(rownames(cor_model)[high_cor[i,1]], "<->",
          colnames(cor_model)[high_cor[i,2]],":",
          round(cor_model[high_cor[i,1], high_cor[i,2]],3), "\n")
    }
  }
} else{
  cat("No pairs exceed |r|>0.8")
} #total_pos and total_neg were removed because they were highly correlated with each other. It was replaced with total_feedback 
# ============================================
# SECTION 14: MODELLING SET UP
# ============================================

control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats= 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final")

train_df <- train_matrix
train_df$success_label <- y_train
test_df <- test_matrix
test_df$success_label <- y_test

metrics_fun <- function(model, test_x, test_y, model_name, features, imbalance){
  pred <- predict(model, test_x)
  prob <- predict(model, test_x, type = "prob")[,"High"]
  cm <- confusionMatrix(pred, test_y, positive = "High")
  auc <- as.numeric(pROC::roc(test_y, prob, quiet=TRUE)$auc)
  y_binary <- ifelse(test_y=="High",1,0)
  pr_auc <- MLmetrics::PRAUC(y_pred= prob, y_true= y_binary)
  tibble(
    Model = model_name,
    Features = features,
    Imbalance_Handling = imbalance,
    Threshold = 0.5,
    Accuracy = round(cm$overall["Accuracy"],3),
    Sensitivity = round(cm$byClass["Sensitivity"],3),
    Specificity = round(cm$byClass["Specificity"],3),
    Precision= round(as.numeric(cm$byClass["Precision"]),3),
    F1= round(as.numeric(cm$byClass["F1"]),3),
    Balanced_Acc = round(cm$byClass["Balanced Accuracy"],3),
    ROC_AUC = round(auc,3),
    PR_AUC = round(as.numeric(pr_auc),3),
    Kappa = round(cm$overall["Kappa"],3)
  )
}

#Confusion matrix 
confusion_flex<- function(model, test_x, test_y, model_name){
  pred <- predict(model,test_x)
  cm <- confusionMatrix(pred, test_y, positive="High")
  cm_table <- as.data.frame.matrix(cm$table) %>% 
    rownames_to_column("Predicted") %>% 
    rename(`Actual: Low` = Low, `Actual: High`= High)
  flextable(cm_table) %>% 
    theme_vanilla() %>% 
    autofit() %>% 
    bold(part= "header") %>% 
    bg(part= "body", bg= "#FDF0E8") %>% 
    set_caption(paste0(model_name, ": Confusion Matrix"))
}

#Threshold selection 
find_threshold <- function(model){
  cv_pred <- model$pred
  roc_obj <- pROC::roc(cv_pred$obs, cv_pred$High, quiet = TRUE)
  pROC::coords(roc_obj, "best", best.method= "youden", ret= "threshold")[[1]]
}

eval_threshold <- function(model, test_data, test_y, thresh, model_name, features, imbalance ){
  prob <- predict(model, test_data, type="prob")[,"High"]
  pred <- factor(ifelse(prob>= thresh, "High", "Low"), levels= c("Low", "High"))
  cm <- confusionMatrix(pred, test_y, positive = "High")
  auc <- as.numeric(pROC::roc(test_y, prob, quiet= TRUE)$auc)
  y_binary <- ifelse(test_y=="High",1,0)
  pr_auc <- MLmetrics::PRAUC(y_pred= prob, y_true= y_binary)
  tibble(
    Model= model_name,
    Features = features,
    Imbalance_Handling = imbalance,
    Threshold = round(thresh,3),
    Accuracy= round(cm$overall["Accuracy"], 3),
    Sensitivity  = round(cm$byClass["Sensitivity"], 3),
    Specificity  = round(cm$byClass["Specificity"], 3),
    Precision= round(as.numeric(cm$byClass["Precision"]),3),
    F1= round(as.numeric(cm$byClass["F1"]),3),
    Balanced_Acc = round(cm$byClass["Balanced Accuracy"], 3),
    ROC_AUC      = round(auc, 3),
    PR_AUC = round(as.numeric(pr_auc),3),
    Kappa        = round(cm$overall["Kappa"], 3)
  )
}
all_results <- tibble ()

#Coefficent analysis helper function 
coef_table <- function(model, model_name, variant){
  broom:: tidy(model$finalModel) %>% 
    mutate(term = gsub("`","", term)) %>% 
    mutate(Model = model_name, Variant = variant) %>% 
    select(Model, Variant, term, estimate, std.error, statistic, p.value) %>% 
    arrange(p.value) %>% 
    mutate(across(c(estimate, std.error, statistic), ~round(.,3)),
           p.value = signif(p.value,3))
}

#ROC function 
plot_roc<- function(model_plain, test_data, test_y, model_name){
  prob_plain <- predict(model_plain, test_data, type = "prob")[,"High"]
  roc_plain <- roc(test_y, prob_plain, quiet = TRUE)
  roc_df <- bind_rows(
    tibble(Treatment = "Plain", FPR = 1- roc_plain$specificities, TPR = roc_plain$sensitivities))
  auc_p<- round(as.numeric(roc_plain$auc),3)
  
  ggplot(roc_df, aes(x=FPR, y=TPR, color = Treatment))+
    geom_line(linewidth=1)+
    geom_abline(linetype= "dashed", color = "#B5A5A5")+
    scale_color_manual(
      values = c("Plain" = "#9B6B7B"),
      labels = c("Plain"= paste0("Plain (AUC=", auc_p, ")")))+
    labs(title = paste0(model_name, ": ROC Curve"),
         x= "False Positive Rate", y= "True Positive Rate", color = NULL)+ 
    coord_equal()+
    theme_sephora()
}

#DeLONG TEST 
delong_test <- function(model_struct, test_struct, model_enr, test_enr, test_y, model_name){
  prob_s <- predict(model_struct, test_struct, type = "prob")[,"High"]
  prob_e <- predict(model_enr, test_enr, type = "prob")[,"High"]
  
  roc_s <- roc(test_y, prob_s, quiet=TRUE)
  roc_e <- roc(test_y, prob_e, quiet=TRUE)
  
  test <- roc.test(roc_s, roc_e, method= "delong")
  
  tibble(
    model = model_name,
    AUC_structured= round(as.numeric(roc_s$auc),3),
    AUC_enriched= round(as.numeric(roc_e$auc),3),
    Delta = round(as.numeric(roc_e$auc- roc_s$auc),3),
    p_value = round(test$p.value,4),
    Significant = ifelse(test$p.value<0.05, "Yes", "No")
  )
}

# ============================================
# SECTION 15.1: BASELINE MODELS LOGISTIC REGRESSION
# ============================================

#Plain 
set.seed(40485968)
m_logistic <- train(success_label~., data = train_df, method = "glm", family = "binomial", trControl = control, metric = "ROC", preProcess = c("center", "scale")) #Showed warnings with doubtful cases 

#With threshold optimisation 
opt_thresh <- find_threshold(m_logistic)
cat("Optimal threshold (Logistic):", round(opt_thresh, 3),"\n")
logistic_threshold <- eval_threshold(m_logistic, test_df, y_test, opt_thresh, "Logistic Regression", "Structured", "Threshold Tuning")

#Examining Coefficients 
log_coefs_plain <- coef_table (m_logistic, "Logistic Regression", "Plain")
log_coef_flextable <- bind_rows(log_coefs_plain) %>% 
  flextable() %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part = "header") %>% 
  set_formatter(p.value =function(x) formatC(x, format = "e", digits = 3)) %>% 
  set_caption("Logistic Regression: Coefficients")
log_coef_flextable

#Confusion Matrix 
confusion_flex(m_logistic, test_df, y_test, "Logistic Regression (Structured)")

#Results 
logistic_plain <- metrics_fun(m_logistic, test_df, y_test, "Logistic Regression", "Structured", "Plain")
logistic_results <- bind_rows(logistic_plain, logistic_threshold)
logistic_results <- logistic_results %>% 
  mutate(Threshold= ifelse(is.na(Threshold), 0.5, Threshold))
all_results<- bind_rows(all_results, logistic_results)
logistic_flextable<- flextable(logistic_results) %>% 
  theme_vanilla() %>% autofit() %>% bold(part = "header") %>% 
  set_caption("Logistic Regression: Plain vs Threshold Tuning")
logistic_flextable

#ROC 
plot_roc(m_logistic, test_df, y_test, "Logistic Regression")

# ============================================
# SECTION 15.2: BASELINE MODELS: ELASTIC NET
# ============================================

#Plain 
set.seed(40485968)
m_enet <- train(success_label~.,data = train_df, method = "glmnet", trControl = control, metric = "ROC", tuneLength = 10, preProcess = c("center", "scale"))

#Metrics 
enet_plain <- metrics_fun(m_enet, test_df, y_test, "Elastic Net", "Structured", "Plain")
opt_thresh_enet <- find_threshold(m_enet)
cat("Optimal threshold(Elastic Net):", round(opt_thresh_enet,3),"\n")
enet_threshold <- eval_threshold(m_enet, test_df, y_test, opt_thresh_enet, "Elastic Net", "Structured", "Threshold Tuning")
enet_results <- bind_rows(enet_plain, enet_threshold)
all_results <- bind_rows(all_results, enet_results)

#Coefficent extraction 
enet_coefs <- function(model, model_name, variant){
  co <- coef(model$finalModel, s= model$bestTune$lambda)
  tibble(
    Model = model_name, 
    Variant = variant,
    term= rownames(co),
    coefficient = round(as.numeric(co),4)
  ) %>% 
    mutate(term = gsub("`", "", term)) %>% 
    filter(term!="(Intercept)") %>% 
    arrange(desc(abs(coefficient)))
}
enet_coef_plain <- enet_coefs(m_enet, "Elastic Net", "Plain")
enet_coefs_table <- bind_rows(enet_coef_plain)

enet_coef_flextable <- flextable(enet_coefs_table) %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part = "header") %>% 
  bg(i = ~ coefficient == 0, bg = "#E2E0DC", part = "body") %>%   
  set_caption("Elastic Net: Coefficients (shrunk toward zero = deprioritised)")
enet_coef_flextable

#Results 
enet_flextable <- flextable(enet_results) %>% 
  theme_vanilla() %>% autofit() %>% bold(part = "header") %>% 
  set_caption("Elastic Net: Plain vs Threshold Tuning")
enet_flextable

#Confusion Matrix 
confusion_flex(m_enet, test_df, y_test, "Elastic Net (Structured)")

#ROC 
plot_roc(m_enet, test_df, y_test, "Elastic Net")

#Coefficient path plot 
enet_fit<- m_enet$finalModel

coef_path <- as.matrix(enet_fit$beta) %>% 
  as.data.frame() %>% 
  rownames_to_column("feature") %>% 
  pivot_longer(-feature, names_to = "step", values_to= "coefficient") %>% 
  mutate(feature= gsub("`","", feature)) %>% 
  left_join(
    tibble(step = paste0("s", seq_along(enet_fit$lambda)-1),
           lambda = enet_fit$lambda),
    by= "step"
  )
best_lambda <- m_enet$bestTune$lambda

label_positions <- coef_path %>%
  filter(lambda == min(lambda)) %>%
  mutate(feature = gsub("`", "", feature))

ggplot(coef_path, aes(x = log(lambda), y = coefficient,
                      group = feature, color = feature)) +
  geom_line(linewidth = 0.7, alpha = 0.9) +
  geom_vline(xintercept = log(best_lambda), linetype = "dashed",
             color = "#333333", linewidth = 0.6) +
  ggrepel::geom_text_repel(
    data = label_positions,
    aes(label = feature, color = feature),
    size = 2.8, hjust = 1, direction = "y",
    nudge_x = -0.3, segment.size = 0.2, max.overlaps = Inf, show.legend = FALSE) +
  scale_color_viridis_d(option = "turbo", guide = "none") +  
  labs(title = "Elastic Net: Coefficient Shrinkage Paths",
       x = "log(lamda): increasing penalty", y = "Coefficient") +
  expand_limits(x = min(log(coef_path$lambda)) - 1.5) +   
  theme_sephora() +
  theme(legend.position = "none") #formatted using AI

# ============================================
# SECTION 15.3: BASELINE MODELS: RANDOM FOREST
# ============================================
#Plain
set.seed(40485968)
m_rf <- train(success_label~., data = train_df, method = "rf", trControl = control, metric = "ROC", importance= TRUE, preProcess = c("center", "scale"))

#Metrics
rf_plain <- metrics_fun(m_rf, test_df, y_test, "Random Forest", "Structured", "Plain")

#Threshold tuning
opt_thres_rf <- find_threshold(m_rf)
cat("Optimal threshold random forest:", round(opt_thres_rf,3), "\n")
rf_threshod <- eval_threshold(m_rf, test_df, y_test, opt_thres_rf, "Random Forest", "Structured", "Threshold Tuning")
rf_results<- bind_rows(rf_plain, rf_threshod)
all_results <- bind_rows(all_results, rf_results)

#Results
rf_flextable <- flextable(rf_results) %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part = "header") %>% 
  set_caption("Random Forest: Plain vs Threshold Tuning")
rf_flextable

#Confusion Matrix 
confusion_flex(m_rf, test_df, y_test, "Random Forest (Structured)")

#OOB error 
rf_final <- m_rf$finalModel
print(rf_final)
oob_error <- rf_final$err.rate[nrow(rf_final$err.rate), "OOB"]
cat("Random Forest OOB error:", round(oob_error,4), "\n")

rf_importance <- importance(rf_final, type =1) %>% 
  as.data.frame() %>% 
  rownames_to_column("feature") %>% 
  rename(importance = MeanDecreaseAccuracy) %>% 
  mutate(feature= gsub("`",'', feature)) %>% 
  arrange(desc(importance))
rf_importance
cor(train_matrix$review_velocity, train_matrix$days_active)

#ROC plot 
plot_roc(m_rf, test_df, y_test, "Random Forest")

#Variable importance 
ggplot(rf_importance, aes(x=reorder(feature, importance), y= importance, fill= importance>0))+
  geom_col(width=0.7)+
  coord_flip()+
  scale_fill_manual(
    values= c("TRUE"= "#9B6B7B", "FALSE" = "#C17B7B"),
    labels = c("TRUE" = "Positive (useful)", "FALSE" = "Negative (noise)"),
    name= NULL
  )+
  labs(title= "RandomForest: Permutation Importance", x= NULL, y= "Importance")+
  theme_sephora()

# ============================================
# SECTION 15.4: BASELINE MODELS: KNN
# ============================================
#Plain
set.seed(40485968)
m_knn<- train(success_label~., data = train_df, method = "kknn", trControl = control, metric = "ROC", tuneLength = 10, preProcess = c("center", "scale"))

knn_plain <- metrics_fun(m_knn, test_df, y_test, "KNN", "Structured", "Plain")
opt_thresh_knn <- find_threshold(m_knn)
cat("Optimal threshold (KNN):", round(opt_thresh_knn, 3), "\n")
knn_threshold <- eval_threshold(m_knn, test_df, y_test, opt_thresh_knn, "KNN", "Structured", "Threshold Tuning")

#Results 
knn_results <- bind_rows(knn_plain, knn_threshold)
all_results <- bind_rows(all_results, knn_results)
knn_flextable<- flextable(knn_results) %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold (part= "header") %>% 
  set_caption("KNN:Plain vs Threshold Tuning")
knn_flextable 

#Confusion Matrix 
confusion_flex(m_knn, test_df, y_test, "KNN (Structured)")

#ROC 
plot_roc(m_knn, test_df, y_test, "KNN")

#Feature level importance 
knn_imp <- varImp(m_knn)$importance %>% 
  rownames_to_column("feature") %>% 
  mutate(feature = gsub("`","", feature))

knn_imp <- knn_imp %>%
  mutate(importance = knn_imp[[2]]) %>%   
  select(feature, importance) %>%
  arrange(desc(importance))
knn_imp

#Separation map 
separation_df <- train_df %>% 
  select(reviews, total_feedback, success_label)

ggplot(separation_df, aes(x= reviews, y= total_feedback, color = success_label))+
  geom_point(alpha=0.5, size=1.5)+
  scale_color_manual(values=c("Low"= "#B5A5A5", "High"="#9B6B7B"))+
  labs(title= "KNN: Class Separation in Feature Space",
       x= "Reviews (scaled)", y="Total Feedback (scaled)", color= "Success label")+
  theme_sephora()

# ============================================
# SECTION 15.5: BASELINE MODELS: SVM (Radial)
# ============================================
#Plain
set.seed(40485968)
m_svm <- train(success_label~., data = train_df, method = "svmRadial", trControl = control, metric = "ROC", tuneLength = 10, preProcess = c("center", "scale"))

svm_plain<- metrics_fun(m_svm, test_df, y_test, "SVM", "Structured", "Plain")
opt_thresh_svm<- find_threshold(m_svm)
cat("Optimal threshold (SVM):", round(opt_thresh_svm,3), "\n")
svm_threshold <- eval_threshold(m_svm,test_df, y_test, opt_thresh_svm, "SVM", "Structured", "Threshold Tuning")
svm_results <- bind_rows(svm_plain, svm_threshold)
all_results <- bind_rows(all_results, svm_results)

svm_flextable<- flextable(svm_results) %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part= "header") %>% 
  set_caption("SVM: Plain vs Threshold Tuning")
svm_flextable 

#Confusion Matrix 
confusion_flex(m_svm, test_df, y_test, "SVM Radial (Structured)")

#ROC
plot_roc(m_svm, test_df, y_test, "SVMRadial")

#Predicted probability distribution by true class
svm_plain_df <- tibble(
  prob=predict(m_svm, test_df, type="prob")[,"High"],
  actual = y_test, panel = "Plain (cut=0.5)", cutoff= 0.5
)

svm_thresh_df<- tibble(
  prob= predict(m_svm, test_df, type= "prob")[,"High"],
  actual= y_test,
  panel= paste0("Threshold (cut=", round(opt_thresh_svm,2), ")"),
  cutoff= opt_thresh_svm
)

# ============================================
# SECTION 15.6: BASELINE MODELS: XGBoost
# ============================================
#Plain
set.seed(40485968)
m_xgb <- train(success_label~., data = train_df, method = "xgbTree", trControl = control, metric = "ROC", tuneLength = 5, verbosity =0, preProcess = c("center", "scale"))

xgb_plain <- metrics_fun(m_xgb, test_df, y_test, "XGBoost", "Structured", "Plain")
opt_thresh_xgb<- find_threshold(m_xgb)
cat("Optimal threshold (XGBoost):", round(opt_thresh_xgb,3), "\n")
xgb_threshold <- eval_threshold(m_xgb, test_df, y_test, opt_thresh_xgb,"XGBoost", "Structured", "Threshold Tuning")
xgb_results <- bind_rows(xgb_plain, xgb_threshold)
all_results <- bind_rows(all_results, xgb_results)

xgb_flextable <- flextable(xgb_results) %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part = "header") %>% 
  set_caption("XGBoost: Plain vs Threshold Tuning")
xgb_flextable 

#Confusion Matrix 
confusion_flex (m_xgb, test_df, y_test, "XGBoost (Structured)")

#ROC
plot_roc(m_xgb, test_df, y_test, "XGBoost")

#Gain based importance 
xgb_importance <- xgb.importance(model= m_xgb$finalModel) %>% 
  as.data.frame() %>% 
  select(feature = Feature, importance = Gain) %>% 
  mutate(feature= gsub("`","",feature)) %>% 
  arrange(desc(importance))

ggplot(xgb_importance, 
       aes(x= reorder(feature, importance), y=importance))+
  geom_col(fill = "#9B6B7B", width = 0.7)+
  coord_flip()+
  labs(title= "XGBoost: Feature Importance (Gain)", x= NULL, y= "Gain")+
  theme_sephora()

# ============================================
# SECTION 15.7: BASELINE MODELS: RESULTS 
# ============================================

model_order <- all_results %>% 
  group_by(Model) %>% 
  summarise(best=max(Balanced_Acc)) %>% 
  arrange(Model, desc(best)) %>% 
  pull(Model)

all_results_sorted <- all_results %>% 
  mutate(Model = factor(Model, levels=model_order)) %>% 
  arrange(Model, desc(Balanced_Acc))

master_flex<- flextable(all_results_sorted) %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part="header") %>% 
  bg(i=~Model == "XGBoost", bg = "#F4C2C2", part = "body") %>%
  bg(i = ~ Model == "Random Forest",       bg = "#FDF0E8", part = "body") %>%
  bg(i = ~ Model == "SVM",                 bg = "#F7E7CE", part = "body") %>%
  bg(i = ~ Model == "Elastic Net",         bg = "#E8D5D5", part = "body") %>%
  bg(i = ~ Model == "Logistic Regression", bg = "#EADDD5", part = "body") %>%
  bg(i = ~ Model == "KNN",                 bg = "#DCE8E8", part = "body") %>%
  set_caption("Structured Baseline: All Models x Imbalance Treatments (sorted by balanced accuracy within model)")
master_flex

#Comparative ROC figure
plot_all_roc <- function(models_list, test_data, test_y){
  roc_df <- map_dfr(names(models_list), function (model_name){
    prob <- predict(models_list[[model_name]], test_data, type = "prob")[,"High"]
    roc_obj <- roc(test_y, prob, quiet=TRUE)
    auc_val <- round(as.numeric(roc_obj$auc),3)
    tibble(Model = paste0(model_name, "(AUC=", auc_val, ")"),
           FPR = 1- roc_obj$specificities,
           TPR = roc_obj$sensitivities
    )
  })
  ggplot(roc_df, aes(x=FPR, y=TPR, color = Model))+
    geom_line(linewidth =1)+
    geom_abline(linetype= "dashed", color ="#B5A5A5")+
    labs(title = "ROC curves: Structured Models",
         x= "False Positive Rate",
         y = "True Positive Rate",
         color = NULL)+
    coord_equal()+
    theme_sephora()
}

models_structured <- list(
  "Logistic Regression" = m_logistic,
  "Elastic Net" = m_enet,
  "Random Forest" = m_rf,
  "KNN" = m_knn,
  "SVM"= m_svm,
  "XGBoost"= m_xgb
)

plot_all_roc(models_structured, test_df, y_test)
# ============================================
# SECTION 16: TOPIC MODELLING
# ============================================

reviews <- reviews %>% 
  mutate(
    review_id= row_number(),
    review_text_clean = review_text %>% 
      str_to_lower() %>% 
      str_replace_all("[^a-z\\s]",'') %>% 
      str_squish())

#Sample for exploration 
set.seed(40485968)
reviews_sample<- reviews %>% 
  filter(!is.na(review_text_clean), review_text_clean !="") %>% 
  sample_n(20000)

#Custom stop words 
custom_stops <- tibble(word= c( "product", "products","skin", "use", "used", "using", "really", "love", "loved", "great", "just", "get", "got", "buy", "bought", "sephora", "review", "ive", "im", "dont", "didnt", "will", "one", "also", "much", "well", "even", "day", "days", "week", "weeks", "month", "months", "time", "little", "bit", "made", "makes", "make", "feel", "like", "feels", "good", "nice", "amazing", "cream", "serum", "moisturizer", "moisturiser", "cleanser", "mask", "makeup", "sunscreen", "oil", "lotion", "toner", "spf", "skin", "face", "eye", "eyes", "lip", "lips", "dry", "oily", "combination", "acne", "pores", "pore", "doesnt"))

#Tokenings cleaned text 
review_tokens <- reviews_sample %>% 
  unnest_tokens(word, review_text_clean) %>% 
  anti_join(stop_words, by="word") %>% 
  anti_join(custom_stops,  by="word") %>% 
  filter(str_length(word)>2)

#Trimming rare words and building DTM 
word_counts <- review_tokens %>%  count(review_id, word)
common_words<- word_counts %>% 
  group_by(word) %>% 
  summarise(total=sum(n), .groups="drop") %>% 
  filter(total>=10) %>% 
  pull(word)

review_dtm<- word_counts %>% 
  filter(word %in% common_words) %>% 
  cast_dtm(review_id, word, n)

cat("DTM dimensions:", dim(review_dtm)[1], "documents x", dim(review_dtm)[2], "terms\n")

#Fitting LDA model to find relevant concepts 
lda_model <- LDA(review_dtm, k=8, control= list(seed=40485968))

lda_topics <- tidy(lda_model, matrix = "beta") %>% 
  group_by(topic) %>% 
  slice_max(beta, n=12) %>% 
  arrange(topic, desc(beta))

print(lda_topics, n=96)


# ============================================
# SECTION 17.1: EXTRACTION FOR MANUAL CODING
# ============================================
set.seed(40485968)
concept_words <- list(
  effectiveness = c("work", "works", "worked", "result", "results", "difference", "effective", "cleared", "improved", "noticed", "nothing"),
  irritation = c("irritate", "breakout", "broke", "sensitive", "gentle", "sting", "burn", "redness", "reaction", "rash"),
  texture = c("texture", "greasy", "sticky", "lightweight", "absorb", "smooth", "thick", "creamy", "watery", "oily"),
  fragrance = c("smell", "scent", "fragrance", "perfume", "odour", "odor", "fragrance-free",  "unscented"),
  value = c("price", "worth", "expensive", "cheap", "value", "money", "afford", "pricey", "overpriced")
)

tag_concepts <- function(txt) {
  txt_l <- str_to_lower(txt)
  sapply(concept_words, function(kw) {
    str_detect(txt_l, paste(kw, collapse = "|"))
  })
}

review_pool <- reviews %>% 
  filter(!is.na(review_text), str_length(review_text)>40) %>% 
  select(review_id, product_id, review_text) %>% 
  slice_sample(n=5000)

tags <- t(sapply(review_pool$review_text, tag_concepts))
review_pool <- bind_cols(review_pool, as_tibble(tags))

per_concept<- 16
sampled <- map_df(names(concept_words), function(cpt){
  review_pool %>% filter(.data[[cpt]]) %>% slice_sample(n=per_concept)
}) %>% distinct(review_id, .keep_all = TRUE)

none_mentioned <- review_pool %>% 
  filter(!effectiveness & !irritation & !texture & !fragrance & !value) %>% 
  slice_sample(n=10)

gold_sample <- bind_rows(sampled, none_mentioned) %>% 
  distinct(review_id, .keep_all = TRUE) %>% 
  slice_sample(prop=1)

cat("Gold sample size:", nrow(gold_sample), "reviews\n")

label_sheet <- gold_sample %>% 
  select(review_id, product_id, review_text) %>% 
  mutate(
    Effectiveness= "",
    Irritation= "",
    Texture = "",
    Fragrance= "",
    Value= "",
    Difficult_flag = "",
    Notes= ""
  )

#wb <- createWorkbook()
#addWorksheet(wb, "Labelling")
#writeData(wb, "Labelling", label_sheet)
#saveWorkbook(wb, "data/goldsample.xlsx", overwrite=TRUE)

# ============================================
# SECTION 17.2: INTRA-RATER CONSISTENCY CHECK
# ============================================
gold <- read.xlsx("data/gold_recovered_FINAL.xlsx")
set.seed(40485968)
recheck<- gold %>% slice_sample(prop=0.20)
recheck_blank <- recheck %>% 
  mutate(Effectiveness = "", Irritation="", Texture="", Fragrance="", Value="")
wb<- createWorkbook()
addWorksheet(wb, "Recheck")
writeData(wb, "Recheck", recheck_blank)
saveWorkbook(wb, "data/goldresample.xlsx", overwrite = TRUE)

recheck_new <- read.xlsx("data/goldresample.xlsx")

concepts<- c("Effectiveness", "Irritation", "Texture", "Fragrance", "Value")

kappa_results <- map_dfr(concepts, function(cpt){
  merged <- recheck %>% 
    select(review_id, orig= all_of(cpt)) %>% 
    inner_join(recheck_new %>% select(review_id, new= all_of(cpt)), by= "review_id")
  
  k <- kappa2(merged %>% select(orig, new))
  tibble(Concept = cpt, Kappa = round(k$value,3), Agreement_pct= round(mean(merged$orig== merged$new)*100,1))
})

kappa_results

# ============================================
# SECTION 18.: ZERO SHOT EXTRACTION
# ============================================
classifier <- transformers$pipeline(
  "zero-shot-classification",
  model= "facebook/bart-large-mnli"
)

mention_hyps <- c(
  effectiveness = "this review discusses whether the product works or its results",
  irritation    = "this review says the product caused irritation, stinging, or redness, or that it was gentle on the skin",
  texture       = "this review discusses the texture, consistency, or how the product feels",
  fragrance     = "this review discusses the smell, scent, or fragrance",
  value         = "this review discusses the price, cost, or whether it is worth the money"
)
not_hyp <- "this review does not discuss this topic"

sent_hyps <- list(
  effectiveness = c(POS = "the product worked and gave good results",
                    NEG = "the product did not work or gave poor results"),
  irritation    = c(POS = "the product was gentle and did not irritate the skin",
                    NEG = "the product irritated the skin or caused a reaction"),
  texture       = c(POS = "the product had a pleasant texture",
                    NEG = "the product had an unpleasant or bad texture"),
  fragrance     = c(POS = "the product had a pleasant smell",
                    NEG = "the product had an unpleasant smell"),
  value         = c(POS = "the product was good value for the money",
                    NEG = "the product was overpriced or not worth it")
)

zeroshot_concept <- function(text, concept, threshold=0.8){
  m_res <- classifier(text, c(mention_hyps[[concept]], not_hyp), multi_label =TRUE)
  m_score <- as.numeric(m_res$scores)[as.character(m_res$labels)== mention_hyps[[concept]]]
  if(length(m_score)==0|| m_score<threshold) return("NM")
  
  sh<- sent_hyps[[concept]]
  s_res <- classifier(text, unname(sh), multi_label=FALSE)
  
  winner <- as.character(s_res$labels)[1]
  names(sh)[unname(sh)==winner]
}

zeroshot_predictions <- gold %>% 
  select(review_id, review_text) %>% 
  mutate(
    Effectiveness = map_chr(review_text, ~zeroshot_concept(.x, "effectiveness")),
    Irritation = map_chr(review_text, ~zeroshot_concept(.x, "irritation")),
    Texture= map_chr(review_text, ~zeroshot_concept(.x, "texture")),
    Fragrance= map_chr(review_text, ~zeroshot_concept(.x, "fragrance")),
    Value= map_chr(review_text, ~zeroshot_concept(.x, "value"))
  )

zeroshot_scores <- map_dfr(concepts, function(cpt){
  truth <- gold[[cpt]]; pred <- zeroshot_predictions[[cpt]]
  classes <- c("NM", "POS", "NEG")
  f1s <- sapply(classes, function(cl){
    tp <- sum(pred== cl & truth == cl)
    fp <- sum(pred==cl & truth!= cl)
    fn <- sum(pred !=cl & truth==cl)
    prec <- if((tp+fp)==0)NA else tp/(tp+fp)
    rec<- if((tp+fn)==0)NA else tp/(tp+fn)
    if(is.na(prec)| is.na(rec)|(prec+rec)==0)return(NA)
    2*prec*rec/(prec+rec)
  })
  tibble(Concept=cpt, Accuracy= round(mean(pred==truth),3), Macro_F1= round(mean(f1s, na.rm=TRUE),3))
})
zeroshot_scores

table(Predicted= zeroshot_predictions$Effectiveness, Actual= gold$Effectiveness)

# ============================================
# SECTION 18.3: IMPLEMENTING ZERO SHOT
# ============================================
#Threshold selection for classification 
mention_scores <- map_dfr(concepts, function(cpt){
  cpt_lower <- str_to_lower(cpt)
  scores <- map_dbl(gold$review_text, function(txt){
    m_res <- classifier(txt, c(mention_hyps[[cpt_lower]], not_hyp), multi_label=TRUE)
    as.numeric(m_res$scores)[as.character(m_res$labels) == mention_hyps[[cpt_lower]]]
  })
  tibble(review_id = gold$review_id, concept = cpt, mention_score = scores)
})

sentiment_labels <- map_dfr(concepts, function(cpt){
  cpt_lower <- str_to_lower(cpt)
  sh <- sent_hyps[[cpt_lower]]
  labels <- map_chr(gold$review_text, function(txt){
    s_res <- classifier(txt, unname(sh), multi_label = FALSE)
    winner <- as.character(s_res$labels)[1]
    names(sh)[unname(sh)== winner]
  })
  tibble (review_id = gold$review_id, concept = cpt,sentiment = labels)
})

classes <- c("NM", "POS", "NEG", "MIX")

threshold_sensitivity <- map_dfr(c(0.70,0.75,0.800,0.85,0.90), function(thresh){
  preds <- mention_scores %>% 
    left_join(sentiment_labels, by = c("review_id","concept")) %>% 
    mutate(pred = ifelse(mention_score <thresh, "NM", sentiment))
  
  per_concept <-map_dfr(concepts, function(cpt){
    truth <- gold[[cpt]]
    pred <- preds %>%  filter(concept ==cpt) %>% arrange(match(review_id, gold$review_id)) %>% pull(pred)
    
    f1s <- sapply(classes, function(cl){
      tp <- sum(pred == cl & truth== cl)
      fp <- sum(pred == cl & truth != cl)
      fn <- sum (pred != cl & truth ==cl)
      prec <- if((tp+fp)==0)NA else tp/(tp+fp)
      rec <- if((tp + fn)== 0)NA else tp/(tp+fn)
      if (is.na(prec)|| is.na(rec)|| (prec+rec)==0) return (NA)
      2*prec*rec / (prec+rec)
    })
    tibble (threshold = thresh, concept = cpt, macro_F1 = round (mean(f1s, na.rm= TRUE),3))
  })
  per_concept
})

threshold_sensitivity

#Since zero shot application on 1mil reviews would be computationally expensive, it is better to pick a certain number of reviews per product and apply it to those many
reviews_per_product <- reviews %>% 
  count(product_id, name="n_reviews")

summary(reviews_per_product$n_reviews)
quantile(reviews_per_product$n_reviews, probs= c(0.5,0.75,0.9,0.95,0.99))

ggplot(reviews_per_product, aes(x=n_reviews))+
  geom_histogram(bins=60, fill = "#C17B7B", color = "#FDF0E8")+
  scale_x_log10()+
  labs(title="Reviews per Product",
       x= "Number of reviews (log scale)", y="Number of Products")+
  theme_sephora()

# Test several caps: how many reviews to process, and what % of products are affected
cap_options <- c(30, 40, 50, 75, 100, 150, 200)

map_dfr(cap_options, function(cap) {
  affected <- mean(reviews_per_product$n_reviews > cap)      # % products trimmed
  total_used <- sum(pmin(reviews_per_product$n_reviews, cap))
  tibble(
    Cap = cap,
    Pct_products_capped = round(affected * 100, 1),
    Reviews_to_process = total_used,
    Pct_of_full = round(total_used / sum(reviews_per_product$n_reviews) * 100, 1)
  )
})

#Prep capped sample for colab extraction
set.seed(40485968)
cap_options <- c(30, 40, 50, 75, 100, 150, 200)

moe_table <- map_dfr(cap_options, function (cap){
  n <- cap
  p <- 0.5
  z <- 1.96
  se <- sqrt (p*(1-p)/n)
  moe <- z*se
  tibble(Cap = cap, SE= round(se, 4), MOE = round (moe, 4))
})
moe_table

cap <- 30
capped_reviews <- reviews %>% 
  filter(!is.na(review_text), str_length(review_text)>10) %>% 
  select(review_id, product_id, review_text) %>% 
  group_by(product_id) %>% 
  slice_sample(n=cap) %>% 
  ungroup()
cat("Products:", n_distinct(capped_reviews$product_id), "\n")
cat("Total reviews to process:", nrow(capped_reviews), "\n")
write_csv(capped_reviews, "data/capped_reviews_short.csv")

concept_labels <- read_csv("data/concept_labels.csv")
concept_cols <- c("effectiveness", "irritation", "texture", "fragrance", "value")
concept_labels <- concept_labels %>% 
  mutate(across(all_of(concept_cols), ~toupper(str_trim(as.character(.)))))

aggregate_concept <- function(df,concept){
  df %>% 
    group_by(product_id) %>% 
    summarise(
      n_rev= n(),
      n_pos= sum(.data[[concept]]== "POS", na.rm=TRUE),
      n_neg= sum(.data[[concept]]=="NEG", na.rm=TRUE),
      n_ment= n_pos + n_neg,
      .groups= "drop"
    ) %>% 
    transmute(
      product_id,
      "{concept}_mention_rate":= n_ment/n_rev,
      "{concept}_net":= ifelse(n_ment>0, (n_pos - n_neg)/ n_ment,0)
    )
}
concept_features <- reduce(
  map(concept_cols, ~aggregate_concept(concept_labels, .x)),
  full_join, by= "product_id"
)
glimpse(concept_features)
summary(concept_features)

# ============================================
# SECTION 19: EDA FOR TEXTUAL FEATURES
# ============================================

#Figure 6: Concept mention rates 
concept_prevalence <- concept_features %>% 
  summarise(across(ends_with("_mention_rate"), mean, na.rm = TRUE)) %>% 
  pivot_longer(everything(), names_to= "concept", values_to= "mean_mention_rate") %>% 
  mutate(concept = str_remove(concept, "_mention_rate"),
         concept = str_to_title(concept),
         concept = fct_reorder(concept, mean_mention_rate))
ggplot(concept_prevalence, aes(x= concept, y = mean_mention_rate, fill = concept))+
  geom_col(width=0.7)+
  geom_text(aes(label= scales::percent(mean_mention_rate, accuracy=0.1)),
            hjust=-0.15, size = 3.3, color= "black")+
  coord_flip()+
  scale_fill_manual(values= c("#E8CFC9", "#D4A5A5", "#C99BA5", "#C17B7B", "#9B6B7B"))+
  scale_y_continuous(labels = scales::percent, expand = expansion(mult= c(0,0.15)))+
  labs(title = "Concept Mention Rates Across Reviews", 
       x= NULL, y= "Mean Mention rate")+
  theme_sephora()+
  theme(legend.position = "none")

#Figure 7: Concept polarity 
concept_sentiment<- concept_features %>% 
  select(product_id, ends_with("_net")) %>% 
  pivot_longer(cols = ends_with("_net"), names_to = "concept", values_to = "net_sentiment") %>% 
  mutate(concept = str_remove(concept, "_net"),
         concept = str_to_title(concept)) %>% 
  filter(net_sentiment !=0)

ggplot(concept_sentiment, aes(x= net_sentiment, fill = concept))+
  geom_density(alpha = 0.6, color = NA)+
  facet_wrap (~concept, scales = "free_y", ncol = 1)+
  geom_vline(xintercept = 0, linetype = "dashed", color = "#666666", linewidth =0.4)+
  scale_fill_manual(values = c ("Effectiveness" = "#9B6B7B", "Fragrance" = "#C17B7B",
                                "Texture" = "#C99BA5", "Value" = "#D4A5A5",
                                "Irritation" = "#E8CFC9"))+
  labs(title = "Concepts Differ in Sentiment Polarity",
       x= "Net sentiment (negative to positive)", y= "Density")+
  theme_sephora()+
  theme(legend.position = "none")

#Figure 8: Concept sentiment polarity based on product success
concept_target <- target_frame %>% 
  left_join(concept_features, by = "product_id") %>% 
  select(product_id, success_label, effectiveness_net, fragrance_net, texture_net, value_net, irritation_net) %>% 
  pivot_longer(cols = ends_with("_net"), names_to = "concept", values_to = "net_sentiment") %>% 
  mutate(concept = str_remove(concept, "_net"),
         concept = str_to_title(concept))

ggplot(concept_target, aes(x= concept, y = net_sentiment, fill = success_label))+
  geom_boxplot(alpha = 0.75, outlier.size=0.5, outlier.alpha =0.2)+
  scale_fill_manual(values = c("Low"= "#B5A5A5", "High"= "#9B6B7B"),
                    name = "Success class")+
  labs(title = "Concept Sentiment Distribution by Success Class",
       x= "Net sentiment (negative to positive)", y= "Density")+
  theme_sephora()+
  theme(legend.position = "bottom")

#Figure 9 
concept_summary <- concept_features %>% 
  summarise(across(ends_with("_mention_rate"), mean, na.rm=TRUE),
            across(ends_with("_net"), ~mean(.[.!=0], na.rm=TRUE))) %>% 
  pivot_longer(everything()) %>% 
  mutate(
    metric= case_when(
      str_ends(name, "_mention_rate")~ "mention_rate",
      str_ends(name, "_net")~"net"
    ),
    concept = str_remove(
      name,
      "_(mention_rate|net)$"
    )
  ) %>% 
  select(-name) %>% 
  pivot_wider(names_from = metric, values_from= value) %>% 
  mutate(
    concept = str_to_title(concept)
  )

ggplot(concept_summary, aes(x= mention_rate, y = net, label= concept))+
  geom_point(size = 4, color = "#9B6B7B", alpha = 0.8)+
  ggrepel::geom_text_repel(size=3.4, color = "black")+
  geom_hline(yintercept = 0, linetype = "dashed", color = "#B5A5A5")+
  scale_x_continuous(labels= scales::percent)+
  labs(title= "Concepts by Prevalence and Sentiment Polarity", x= "Mean mention rate", y= "Mean net sentiment (when mentioned)")+
  theme_sephora()

#Figure 10 
cor_frame <- analysis_products %>% 
  left_join(concept_features, by = "product_id") %>% 
  select(rating, loves_count, reviews, total_feedback, price_usd, percent_recommended, days_active, review_velocity, effectiveness_net, fragrance_net, value_net, texture_net, irritation_net) %>% 
  drop_na() 

cor_matrix <- cor(cor_frame, use= "complete.obs")
par(mar = c(1, 1, 6, 1))
corrplot(cor_matrix, method = "color", type = "upper", col = colorRampPalette(c("#6B4C57", "#FDF0E8", "#9B6B7B"))(200),
         tl.col = "black", tl.cex=0.7, addCoef.col = "black", number.cex = 0.5,
         title = "Correlation Among Structured and Concept Features",
         mar = c(0, 0, 2, 0))
# ============================================
# SECTION 20: ENRICHED FRAME CONSTRUCTION
# ============================================

#Rebuild predictor frames 
train_x_enr <- train_raw %>% 
  select(-any_of(c(setdiff(firewall, "product_id"), drop_cols, "success_label"))) %>% 
  mutate(across(where(is.factor), droplevels))
test_x_enr <- test_raw %>% 
  select(-any_of(c(setdiff(firewall, "product_id"), drop_cols, "success_label"))) %>% 
  mutate(across(where(is.factor), droplevels))

#Joining product features by product id
train_x_enr <- train_x_enr %>%  left_join(concept_features, by= "product_id")
test_x_enr <- test_x_enr %>%  left_join(concept_features, by= "product_id")

#handling products with no concept features 
concept_feature_names <- concept_features %>%  select(-product_id) %>%  names()
train_x_enr <- train_x_enr %>% 
  mutate(across(all_of(concept_feature_names), ~replace_na(.,0)))
test_x_enr <- test_x_enr %>% 
  mutate(across(all_of(concept_feature_names), ~replace_na(.,0)))

#Dropping product_id 
train_x_enr <- train_x_enr %>% select(-product_id)
test_x_enr <- test_x_enr %>% select(-product_id)

#Rare level collapse 
train_x_enr <- collapse_rare(train_x_enr, rare_levels)
test_x_enr <- collapse_rare(test_x_enr, rare_levels)
test_x_enr$secondary_category <- factor(test_x_enr$secondary_category, levels = levels(train_x_enr$secondary_category))

#Dummy encoding and preprocessing 
dummy_fun_enr <- dummyVars("~.", data = train_x_enr, fullRank = TRUE )
train_matrix_enr <- as.data.frame(predict(dummy_fun_enr, train_x_enr))
test_matrix_enr <- as.data.frame(predict(dummy_fun_enr, test_x_enr))

#Assembling modelling frames 
train_df_enr <- train_matrix_enr
train_df_enr$success_label <- y_train
test_df_enr <- test_matrix_enr
test_df_enr$success_label <- y_test

#Verification checks 
cat("Enriched train:", dim(train_df_enr)[1], "x", dim(train_df_enr)[2],"\n")
cat("Structured train:", dim(train_df)[1], "x", dim(train_df)[2], "\n")
cat("Extra columns:", ncol(train_df_enr)- ncol(train_df), "\n")
cat("Concept features present:", all(concept_feature_names %in% names(train_df_enr)), "\n")
cat("Same row count as structured:", nrow(train_df_enr)== nrow(train_df), nrow(test_df_enr)==nrow(test_df), "\n")
cat("y aligned:", length(y_train)== nrow(train_df_enr), "\n")

all_results_enr <- tibble()  
# ============================================
# SECTION 21.1:ENRICHED MODELS: LOGISTIC REGRESSION
# ============================================
#Plain
set.seed(40485968)
m_logistic_enr <- train(success_label~., data = train_df_enr, method = "glm", family = "binomial", trControl= control, metric="ROC", preProcess = c("center", "scale"))

#Threshold Optimisation 
opt_thresh_log_enr <- find_threshold(m_logistic_enr)

#Results
log_enr_plain <- metrics_fun(m_logistic_enr, test_df_enr, y_test, "Logistic Regression", "Enriched", "Plain")
log_enr_threshold <- eval_threshold(m_logistic_enr, test_df_enr, y_test, opt_thresh_log_enr, "Logistic Regression", "Enriched", "Threshold Tuning")
all_results_enr <- bind_rows(all_results_enr, log_enr_plain, log_enr_threshold)

#Confusion Matrix 
confusion_flex(m_logistic_enr, test_df_enr, y_test, "Logistic Regression (Enriched)")

#Coefficient analysis
log_enr_coefs_plain <- coef_table(m_logistic_enr, "Logisitic Regression (Enriched)", "Plain")
log_enr_coef_flex <- bind_rows(log_enr_coefs_plain) %>% 
  mutate(is_concept= str_detect(term, "_net|_mention_rate")) %>% 
  flextable() %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part="header") %>% 
  bold(i= ~is_concept ==TRUE, part= "body") %>% 
  set_formatter(p.value = function(x) formatC(x, format= "e", digits = 3)) %>% 
  set_caption("Enriched Logistic: Coefficients (concept features bolded)")
log_enr_coef_flex

plot_roc(m_logistic_enr, test_df_enr, y_test, "Logistic Regression (Enriched)")

# ============================================
# SECTION 21.2:ENRICHED MODELS: ELASTIC NET 
# ============================================
#Plain 
set.seed(40485968)
m_enet_enr <- train(success_label~., data= train_df_enr, method= "glmnet", trControl = control, metric= "ROC", tuneLength= 10,preProcess = c("center", "scale"))

#Threshold optimisation 
opt_thresh_enet_enr <- find_threshold(m_enet_enr)

#Results
enet_enr_plain <- metrics_fun(m_enet_enr, test_df_enr, y_test, "Elastic Net", "Enriched", "Plain")
enet_enr_threshold <- eval_threshold(m_enet_enr, test_df_enr, y_test, opt_thresh_enet_enr, "Elastic Net", "Enriched", "Threshold Tuning")
all_results_enr <- bind_rows(all_results_enr, enet_enr_plain, enet_enr_threshold)

#Confsuion Matrix 
confusion_flex(m_enet_enr, test_df_enr, y_test, "Elastic Net (Enriched)")

#Coefficient analysis 
enet_enr_coef_plain <- enet_coefs (m_enet_enr, "Elastic Net (Enriched)", "Plain")
enet_enr_coef_flex <- bind_rows(enet_enr_coef_plain) %>% 
  mutate(is_concept = str_detect(term, "_net|_mention_rate")) %>% 
  flextable() %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part= "header") %>% 
  bg(i = ~ coefficient == 0, bg = "#E2E0DC", part = "body") %>%
  bold(i = ~ is_concept == TRUE, part = "body") %>%
  set_caption("Enriched Elastic Net: Coefficients (concepts bolded; grey = shrunk to 0)")
enet_enr_coef_flex

#ROC
plot_roc(m_enet_enr, test_df_enr, y_test, "Elastic Net (Enriched)")

#Coefficient path plot 
enet_enr_fit <- m_enet_enr$finalModel
coef_path_enr <- as.matrix(enet_enr_fit$beta) %>% 
  as.data.frame() %>% 
  rownames_to_column("feature") %>% 
  pivot_longer(-feature, names_to= "step", values_to= "coefficient") %>% 
  mutate(feature = gsub("`", "", feature)) %>% 
  left_join(tibble(step= paste0("s", seq_along(enet_enr_fit$lambda)-1),
                   lambda= enet_enr_fit$lambda), by= "step")
best_lambda_enr <- m_enet_enr$bestTune$lambda
label_positions_enr <- coef_path_enr %>% 
  filter(lambda== min(lambda)) %>% 
  mutate(feature = gsub("`", "", feature),
         is_concept= str_detect(feature, "_net|_mention_rate"))

ggplot(coef_path_enr, aes(x= log(lambda), y = coefficient, group = feature, color = feature))+
  geom_line(linewidth= 0.7, alpha = 0.9)+
  geom_vline(xintercept = log(best_lambda_enr), linetype= "dashed", color = "#333333", linewidth = 0.6)+
  ggrepel:: geom_text_repel(data = label_positions_enr, aes(label= feature, color = feature), 
                            size =2.8, hjust=1, direction = "y", nudge_x=- 0.3, segment.size =0.2,max.overlaps= Inf, show.legend = FALSE)+
  scale_color_viridis_d(option = "turbo", guide= "none")+
  labs(title= "Enriched Elastic Net: Coefficient Shrinkage Paths", x= "log Lambda", y = "Coefficient")+
  expand_limits(x= min(log(coef_path_enr$lambda))-1.5)+
  theme_sephora()+
  theme(legend.position = "none")

# ============================================
# SECTION 21.3:ENRICHED MODELS: RANDOM FOREST
# ============================================
#Plain 
set.seed(40485968)
m_rf_enr <- train(success_label~., data= train_df_enr, method = "rf", trControl = control, metric = "ROC", importance= TRUE, preProcess = c("center", "scale"))

#Threshold optimisation 
opt_thresh_rf_enr <- find_threshold(m_rf_enr)

#Results 
rf_enr_plain <- metrics_fun(m_rf_enr, test_df_enr, y_test, "Random Forest", "Enriched", "Plain")
rf_enr_threshold <-  eval_threshold(m_rf_enr, test_df_enr, y_test, opt_thresh_rf_enr, "Random Forest", "Enriched", "Threshold Tuning")
all_results_enr <- bind_rows(all_results_enr, rf_enr_plain, rf_enr_threshold)

#Confsuion Matrix 
confusion_flex(m_rf_enr, test_df_enr, y_test, "Random Forest (Enriched)")

#OOB
rf_enr_final <- m_rf_enr$finalModel 
print(rf_enr_final)
oob_error_enr <- rf_enr_final$err.rate[nrow(rf_enr_final$err.rate), "OOB"]
cat("Enriched RF OOB error:", round(oob_error_enr,4), "\n")

#RF importance 
rf_enr_importance <- importance(rf_enr_final, type=1) %>% 
  as.data.frame() %>% 
  rownames_to_column("feature") %>% 
  rename(importance= MeanDecreaseAccuracy) %>% 
  mutate(feature = gsub("`","", feature),
         is_concept= str_detect(feature, "_net|_mention_rate")) %>% 
  arrange(desc(importance))
rf_enr_importance

ggplot(rf_enr_importance, aes(x= reorder(feature, importance), y= importance, fill = is_concept))+
  geom_col(width= 0.7)+ coord_flip()+
  scale_fill_manual(values = c("FALSE" = "#C17B7B", "TRUE" = "#9B6B7B"),
                    labels = c("FALSE" = "Structured", "TRUE" = "Concept"), name = NULL) +
  labs(title = "Enriched Random Forest: Permutation Importance", x=NULL, y = "Importance")+
  theme_sephora()

#ROC
plot_roc(m_rf_enr, test_df_enr, y_test, "Random Forest (Enriched)")

# ============================================
# SECTION 21.4:ENRICHED MODELS: KNN
# ============================================
#Plain 
set.seed(40485968)
m_knn_enr <- train(success_label ~., data = train_df_enr, method = "kknn", trControl = control, metric = "ROC", tuneLength= 10, preProcess = c("center", "scale"))

#Threshold Optimisation 
opt_thresh_knn_enr <- find_threshold(m_knn_enr)

#Results 
knn_enr_plain <- metrics_fun(m_knn_enr, test_df_enr, y_test, "KNN", "Enriched", "Plain")
knn_enr_threshold <- eval_threshold(m_knn_enr, test_df_enr, y_test, opt_thresh_knn_enr,
                                    "KNN", "Enriched", "Threshold Tuning")
all_results_enr <- bind_rows(all_results_enr, knn_enr_plain, knn_enr_threshold)

#Confusion Matrix 
confusion_flex(m_knn_enr, test_df_enr, y_test, "KNN (Enriched)")

#ROC
plot_roc(m_knn_enr, test_df_enr, y_test, "KNN(Enriched)")

#Importance
knn_enr_imp <- varImp(m_knn_enr)$importance %>% 
  rownames_to_column("feature") %>% 
  mutate(feature = gsub("`", "", feature))
knn_enr_imp <- knn_enr_imp %>% 
  mutate(importance= knn_enr_imp[[2]]) %>% 
  select(feature,importance) %>% 
  arrange(desc(importance))
knn_enr_imp

#Separation map 
separation_enr_df <- train_df_enr %>% 
  select(effectiveness_net, reviews, success_label)
ggplot(separation_enr_df, aes(x= reviews, y= effectiveness_net, color = success_label))+
  geom_point(alpha=0.5, size = 1.5)+
  scale_color_manual(values = c("Low" = "#B5A5A5", "High" = "#9B6B7B")) +
  labs(title = "Enriched KNN: Class Separation (concept vs engagement axis)", x= "Reviews (scaled)", y="Effectiveness net sentiment(scaled)", color = "Success label")+
  theme_sephora()

# ============================================
# SECTION 21.5:ENRICHED MODELS: SVM (Radial)
# ============================================
#Plain 
set.seed(40485968)
m_svm_enr <- train(success_label~., data = train_df_enr, method = "svmRadial", trControl= control, metric = "ROC", tuneLength = 10, preProcess = c("center", "scale"))


#Threshold Optimisation 
opt_thresh_svm_enr <- find_threshold(m_svm_enr)

#Results
svm_enr_plain <- metrics_fun(m_svm_enr, test_df_enr, y_test, "SVM", "Enriched", "Plain")
svm_enr_threshold <- eval_threshold(m_svm_enr, test_df_enr, y_test, opt_thresh_svm_enr, "SVM", "Enriched", "Threshold Tuning")
all_results_enr <- bind_rows(all_results_enr, svm_enr_plain, svm_enr_threshold)

#Confsuion Matrix
confusion_flex(m_svm_enr, test_df_enr, y_test, "SVM Radial (Enriched)")

#ROC
plot_roc(m_svm_enr, test_df_enr, y_test, "SVM (Enriched)")

#Probability distribution 
svm_enr_plain_df <- tibble(prob = predict(m_svm_enr, test_df_enr, type = "prob")[,"High"],
                           actual = y_test, panel = "Plain (cut=0.5)", cutoff = 0.5)
svm_enr_thresh_df <- tibble(prob = predict(m_svm_enr, test_df_enr, type = "prob")[,"High"],
                            actual = y_test,
                            panel = paste0("Threshold (cut=", round(opt_thresh_svm_enr, 2), ")"),
                            cutoff = opt_thresh_svm_enr)
svm_enr_all <- bind_rows(svm_enr_plain_df, svm_enr_thresh_df) %>%
  mutate(panel = factor(panel, levels = c("Plain (cut=0.5)",
                                          unique(svm_enr_thresh_df$panel))))
cutoffs_enr <- svm_enr_all %>% distinct(panel, cutoff)

ggplot(svm_enr_all, aes(x= prob, fill = actual))+
  geom_density(alpha=0.5)+
  geom_vline(data=cutoffs_enr, aes(xintercept= cutoff), linetype= "dashed", color = "#9B6B7B")+
  facet_wrap(~panel, ncol=2)+
  scale_fill_manual(values = c("Low" = "#B5A5A5", "High" = "#9B6B7B"))+
  labs(title = "Enriched SVM: Probability Distribution Across Imbalance Treatment", x= "Predicted Probability of High", y= "Density", fill = "Actual")+
  theme_sephora()+
  theme(legend.position= "bottom")

# ============================================
# SECTION 21.6:ENRICHED MODELS: XGBOOST
# ============================================
#Plain
set.seed(40485968)
m_xgb_enr <- train(success_label~., data= train_df_enr, method = "xgbTree", trControl= control, metric= "ROC", tuneLength=5, verbosity = 0, preProcess = c("center", "scale"))

#Threshold Optimisation 
opt_thresh_xgb_enr <- find_threshold(m_xgb_enr)

#Results <- 
xgb_enr_plain <- metrics_fun(m_xgb_enr, test_df_enr, y_test, "XGBoost", "Enriched", "Plain")
xgb_enr_threshold <- eval_threshold(m_xgb_enr, test_df_enr, y_test, opt_thresh_xgb_enr, "XGBoost", "Enriched", "Threshold Tuning")
all_results_enr <- bind_rows(all_results_enr, xgb_enr_plain, xgb_enr_threshold)

#Confusion Matrix 
confusion_flex(m_xgb_enr, test_df_enr, y_test, "XGBosot (Enriched)")

#ROC
plot_roc(m_xgb_enr, test_df_enr, y_test, "XGBoost (Enriched)")

#XGBoost Gain 
xgb_enr_importance <- xgb.importance(model= m_xgb_enr$finalModel) %>% 
  as.data.frame() %>% 
  select(feature= Feature, importance= Gain) %>% 
  mutate(feature= gsub("`","",feature),
         is_concept= str_detect(feature, "_net|_mention_rate")) %>% 
  arrange(desc(importance))

ggplot(xgb_enr_importance, aes(x=reorder(feature, importance), y= importance, fill= is_concept))+
  geom_col(width=0.7)+
  coord_flip()+
  scale_fill_manual(values = c("FALSE" = "#B5A5A5", "TRUE" = "#9B6B7B"),
                    labels = c("FALSE" = "Structured", "TRUE" = "Concept"),
                    name = NULL) +
  labs(title = "Enriched XGBoost: Feature Importance (Gain)", x= NULL, y="Gain")+
  theme_sephora()

save.image("outputs/dissertation_full.RData")
# ============================================
# SECTION 22 DeLONG TEST
# ============================================
delong_results <- bind_rows (
  delong_test(m_logistic, test_df, m_logistic_enr, test_df_enr, y_test, "Logistic Regression"),
  delong_test(m_enet, test_df, m_enet_enr, test_df_enr, y_test, "Elastic Net"),
  delong_test(m_rf, test_df, m_rf_enr, test_df_enr, y_test, "Random Forest"),
  delong_test(m_knn, test_df, m_knn_enr, test_df_enr, y_test, "KNN"),
  delong_test(m_svm, test_df, m_svm_enr, test_df_enr, y_test, "SVM"),
  delong_test(m_xgb, test_df, m_xgb_enr, test_df_enr, y_test, "XGBoost")
)

delong_flex <- flextable (delong_results) %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part= "header") %>% 
  bg(i = ~ Significant == "Yes", bg = "#F4C2C2", part = "body") %>%   
  color(i = ~ Delta > 0, j = "Delta", color = "#2E7D32", part = "body") %>%  
  color(i = ~ Delta < 0, j = "Delta", color = "#C62828", part = "body") %>%  
  set_caption ("DeLong Test: Structured vs Enriched AUC (per model)")

delong_flex

# ============================================
# SECTION 23 STRUCTURED vs ENRICHED DELTA ANALYSIS
# ============================================

comparison_table <- all_results %>% 
  select(Model, Imbalance_Handling, Features, ROC_AUC, PR_AUC, F1, Balanced_Acc) %>% 
  pivot_wider(
    names_from = Features,
    values_from = c(ROC_AUC, PR_AUC, F1, Balanced_Acc)) %>% 
  mutate(AUC_delta = ROC_AUC_Enriched - ROC_AUC_Structured,
         PR_AUC_Delta = PR_AUC_Enriched - PR_AUC_Structured,
         F1_delta = F1_Enriched - F1_Structured,
         BalAcc_delta= Balanced_Acc_Enriched - Balanced_Acc_Structured) %>% 
  arrange(Model, Imbalance_Handling)

comparison_table 

comparison_flex <- comparison_table %>% 
  select(Model, Imbalance_Handling, ROC_AUC_Structured, ROC_AUC_Enriched, AUC_delta, PR_AUC_Structured, PR_AUC_Enriched, PR_AUC_Delta) %>% 
  flextable() %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part= "header") %>% 
  color(i = ~ AUC_delta > 0, j = "AUC_delta", color = "#2E7D32") %>%
  color(i = ~ AUC_delta < 0, j = "AUC_delta", color = "#C62828") %>%
  color(i = ~ PR_AUC_Delta > 0, j = "PR_AUC_Delta", color = "#2E7D32") %>%
  color(i = ~ PR_AUC_Delta < 0, j = "PR_AUC_Delta", color = "#C62828") %>%
  set_caption("Structured vs Enriched: Metric Deltas by Model Treatment")
comparison_flex
save.image("outputs/dissertation_full.RData")

model_order_enr <- all_results_enr %>% 
  group_by(Model) %>% 
  summarise(best=max(Balanced_Acc)) %>% 
  arrange(Model, desc(best)) %>% 
  pull(Model)

all_results_enr_sorted <- all_results_enr %>% 
  mutate(Model = factor(Model, levels=model_order)) %>% 
  arrange(Model, desc(Balanced_Acc))

master_flex_enr<- flextable(all_results_enr_sorted) %>% 
  theme_vanilla() %>% 
  autofit() %>% 
  bold(part="header") %>% 
  bg(i=~Model == "XGBoost", bg = "#F4C2C2", part = "body") %>%
  bg(i = ~ Model == "Random Forest",       bg = "#FDF0E8", part = "body") %>%
  bg(i = ~ Model == "SVM",                 bg = "#F7E7CE", part = "body") %>%
  bg(i = ~ Model == "Elastic Net",         bg = "#E8D5D5", part = "body") %>%
  bg(i = ~ Model == "Logistic Regression", bg = "#EADDD5", part = "body") %>%
  bg(i = ~ Model == "KNN",                 bg = "#DCE8E8", part = "body") %>%
  set_caption("Enriched Models: All Models x Imbalance Treatments (sorted by balanced accuracy within model)")
master_flex_enr


#Comparative ROC figure
plot_all_roc_enr <- function(models_list, test_data, test_y){
  roc_df <- map_dfr(names(models_list), function (model_name){
    prob <- predict(models_list[[model_name]], test_data, type = "prob")[,"High"]
    roc_obj <- roc(test_y, prob, quiet=TRUE)
    auc_val <- round(as.numeric(roc_obj$auc),3)
    tibble(Model = paste0(model_name, "(AUC=", auc_val, ")"),
           FPR = 1- roc_obj$specificities,
           TPR = roc_obj$sensitivities
    )
  })
  ggplot(roc_df, aes(x=FPR, y=TPR, color = Model))+
    geom_line(linewidth =1)+
    geom_abline(linetype= "dashed", color ="#B5A5A5")+
    labs(title = "ROC urves: Enriched Models",
         x= "False Positive Rate",
         y = "True Positive Rate",
         color = NULL)+
    coord_equal()+
    theme_sephora()
}

models_enr <- list(
  "Logistic Regression" = m_logistic_enr,
  "Elastic Net" = m_enet_enr,
  "Random Forest" = m_rf_enr,
  "KNN" = m_knn_enr,
  "SVM"= m_svm_enr,
  "XGBoost"= m_xgb_enr
)

plot_all_roc_enr(models_enr, test_df_enr, y_test)

# ============================================
# SECTION 24: SHAP ANALYSIS: XGBOOST ENRICHED
# ============================================
X_enr <- as.matrix(train_df_enr %>% select(-success_label))
colnames(X_enr)<- m_xgb_enr$finalModel$feature_names

shap_result <- shap.values(xgb_model= m_xgb_enr$finalModel, X_train= X_enr)
shap_high <- shap_result
shap_high$shap_score <- -shap_result$shap_score
shap_high$BIAS0 <- -shap_result$BIAS0
shap_high$mean_shap_score <- shap_result$mean_shap_score

shap_long_high <- shap.prep(xgb_model = m_xgb_enr$finalModel, X_train = X_enr)
shap_long_high$value <- - shap_long_high$value

shap.plot.summary(shap_long_high)+
  labs(title = "SHAP summary (oriented towards High Success)", x= "SHAP value")

concepts_shap <- c("effectiveness_net", "fragrance_net", "value_net", "texture_net", "irritation_net")
shap_signed <- as.data.frame(shap_high$shap_score)
X_enr_df <- as.data.frame(X_enr)


direction_high <- sapply(concepts_shap, function(f){
  cor(X_enr_df[[f]], shap_high$shap_score[[f]], use = "complete.obs")
})
direction_high

shap.plot.summary(shap_long_high)+
  labs(
    title = "SHAP Summary: Drivers of High-Success Predictions", x= "SHAP value"
  )
save.image("outputs/dissertation_full.RData")

# ============================================
# SECTION 25: COUNTERFACTUALS
# ============================================
test_probs <- predict(m_xgb_enr, test_df_enr, type= "prob")[,"High"]
test_pred <- ifelse(test_probs>=0.5, "High", "Low")
low_idx <- which(test_pred=="Low")
cat("Products predicted low:", length(low_idx), "\n")

counterfactual_combined <- function (product_row, model, step =0.1, max_val = 1){
  original_prob <- predict( model, product_row, type = "prob")[,"High"]
  eff_0 <- product_row[["effectiveness_net"]]
  fra_0 <- product_row[["fragrance_net"]]
  for(delta in seq(0,max_val, by = step)){
    modified <- product_row
    modified[["effectiveness_net"]] <- min(eff_0+delta, max_val)
    modified[["fragrance_net"]]<- min(fra_0+delta, max_val)
    new_prob <- predict(model, modified, type = "prob")[,"High"]
    if(new_prob>=0.5){
      return(tibble(original_prob= round(original_prob,3),
                    delta_applied = round(delta,3),
                    flip_prob = round(new_prob,3), flipped = TRUE))
    }
  }
  tibble(original_prob = round(original_prob,3), delta_applied= NA_real_, flip_prob = NA_real_, flipped= FALSE)
}

borderline_low <- which(test_pred == "Low" & test_probs > 0.35 & test_probs < 0.5)

cf_combined <- map_dfr(borderline_low, function(i){
  counterfactual_combined(test_df_enr[i,, drop = FALSE], m_xgb_enr) %>% 
    mutate(product_idx=i)
})

cf_combined %>% summarise(
  n = n(),
  n_flipped = sum(flipped),
  pct_flipped = round(mean(flipped)*100,1),
  median_delta = median(delta_applied, na.rm=TRUE),
  mean_delta = round(mean(delta_applied, na.rm = TRUE),3)
)

cf_all_low <- map_dfr(low_idx, function(i){
  counterfactual_combined(test_df_enr[i,, drop = FALSE],m_xgb_enr) %>% 
    mutate(product_idx = i, start_prob = test_probs[i])
})

cf_all_low %>% summarise(
  n= n(),
  n_flipped = sum(flipped),
  pct_flipped = round(mean(flipped)*100,1),
  median_delta = median(delta_applied, na.rm = TRUE),
  mean_delta = round(mean(delta_applied, na.rm=TRUE),3))

cf_all_low %>% 
  mutate(
    prob_band = cut(start_prob, 
                    breaks= c(0,0.2,0.35,0.5),
                    labels = c("Deep Low (0-0.2)", "Mid Low (0.2-0.35)", "Borderline (0.35-0.50)"))) %>% 
  group_by(prob_band) %>% 
  summarise(
    n= n(),
    n_flipped = sum(flipped),
    pct_flipped = round(mean(flipped)*100,1),
    median_delta = round(median(delta_applied, na.rm= TRUE),3),
    .groups = "drop"
  )
cf_banded <- cf_all_low %>% 
  mutate(prob_band = cut(start_prob,
                         breaks = c(0, 0.2, 0.35, 0.5),
                         labels = c("Deep Low (0-0.2)", "Mid Low (0.2-0.35)", "Borderline (0.35-0.5)")))

cf_banded %>% 
  group_by(prob_band) %>% 
  summarise(
    n= n(),
    n_flipped = sum(flipped),
    pct_flipped = round(mean(flipped)*100,1),
    median_delta = round(median(delta_applied, na.rm = TRUE),3),
    .groups= "drop"
  ) %>% 
  print(Inf)
cf_banded %>%
  group_by(prob_band) %>%
  summarise(pct_flipped = round(mean(flipped) * 100, 1), .groups = "drop") %>%
  ggplot(aes(x = prob_band, y = pct_flipped, fill = prob_band)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(pct_flipped, "%")), vjust = -0.5) +
  scale_fill_manual(values = c("#B5A5A5", "#C17B7B", "#9B6B7B")) +
  labs(title = "Counterfactual Flip Rate by Distance from Decision Boundary",
       subtitle = "Combined effectiveness + fragrance improvement; structure held fixed",
       x = NULL, y = "% flipped to High") +
  theme_sephora() + theme(legend.position = "none")
save.image("outputs/dissertation_full.RData")

#Significance test for PRAUC
pr_auc_boot_test <- function(model_struct, test_struc, model_enr, test_enr, test_y,model_name, n_boot=2000, seed= 40485968){
  prob_s <- predict(model_struct, test_struc, type= "prob")[,"High"]
  prob_e <- predict(model_enr, test_enr, type = "prob")[, "High"]
  y_bin <- ifelse(test_y== "High", 1,0)
  n <- length(y_bin)
  
  pr_s_obs <-MLmetrics::PRAUC(prob_s, y_bin)
  pr_e_obs <- MLmetrics::PRAUC(prob_e, y_bin)
  obs_diff <- pr_e_obs - pr_s_obs 
  
  set.seed(seed)
  boot_diffs <- numeric(n_boot)
  for(b in seq_len(n_boot)){
    idx <- sample.int(n,n, replace= TRUE)
    if(length(unique(y_bin[idx]))<2){boot_diffs[b]<-NA; next}
    pr_s <- MLmetrics::PRAUC(prob_s[idx], y_bin[idx])
    pr_e <- MLmetrics::PRAUC(prob_e[idx], y_bin[idx])
    boot_diffs[b]<-pr_e-pr_s
  }
  boot_diffs <- boot_diffs[!is.na(boot_diffs)]
  ci <- quantile(boot_diffs, c(0.025,0.975))
  p_boot <- 2*min(mean(boot_diffs<=0), mean(boot_diffs>=0))
  p_boot <- min(p_boot,1)
  
  tibble(
    Model = model_name, 
    PRAUC_structured = round(pr_s_obs,3),
    PRAUC_enriched = round(pr_e_obs,3),
    Delta = round(obs_diff,3),
    CI_lower = round(as.numeric(ci[1]),3),
    CI_upper = round(as.numeric(ci[2]),3),
    p_value = round(p_boot,4),
    Significant = ifelse(ci[1]>0| ci[2]<0, "Yes", "No")
  )
}

prauc_significance <- bind_rows(
  pr_auc_boot_test(m_logistic, test_df, m_logistic_enr, test_df_enr, y_test, "Logistic Regression"),
  pr_auc_boot_test(m_enet, test_df, m_enet_enr, test_df_enr, y_test, "Elastic Net"),
  pr_auc_boot_test(m_rf, test_df, m_rf_enr, test_df_enr, y_test,"Random Forest"),
  pr_auc_boot_test(m_knn, test_df, m_knn_enr, test_df_enr, y_test, "KNN"),
  pr_auc_boot_test(m_svm, test_df, m_svm_enr, test_df_enr, y_test, "SVM"),
  pr_auc_boot_test(m_xgb, test_df, m_xgb_enr, test_df_enr, y_test, "XGBoost")
)

prauc_significance %>% 
  print(Inf)#Generated using AI and edited to fit the purposes of this project 

cm_t <- function(model, test_data, thresh, name){
  prob <- predict(model, test_data, type = "prob")[, "High"]
  pred <- factor(ifelse(prob >= thresh, "High", "Low"), levels = c("Low", "High"))
  cat("\n====", name, "| cut-off =", round(thresh, 3), "====\n")
  print(table(Predicted = pred, Actual = y_test))
}

cm_t(m_logistic,     test_df,     opt_thresh,          "Logistic Regression (Structured)")
cm_t(m_enet,         test_df,     opt_thresh_enet,     "Elastic Net (Structured)")
cm_t(m_rf,           test_df,     opt_thres_rf,        "Random Forest (Structured)")
cm_t(m_knn,          test_df,     opt_thresh_knn,      "KNN (Structured)")
cm_t(m_svm,          test_df,     opt_thresh_svm,      "SVM (Structured)")
cm_t(m_xgb,          test_df,     opt_thresh_xgb,      "XGBoost (Structured)")
cm_t(m_logistic_enr, test_df_enr, opt_thresh_log_enr,  "Logistic Regression (Enriched)")
cm_t(m_enet_enr,     test_df_enr, opt_thresh_enet_enr, "Elastic Net (Enriched)")
cm_t(m_rf_enr,       test_df_enr, opt_thresh_rf_enr,   "Random Forest (Enriched)")
cm_t(m_knn_enr,      test_df_enr, opt_thresh_knn_enr,  "KNN (Enriched)")
cm_t(m_svm_enr,      test_df_enr, opt_thresh_svm_enr,  "SVM (Enriched)")
cm_t(m_xgb_enr,      test_df_enr, opt_thresh_xgb_enr,  "XGBoost (Enriched)")
