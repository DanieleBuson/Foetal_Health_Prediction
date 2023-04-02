df_fetal_health <- read.csv("../Data/fetal_health.csv")
head(df_fetal_health)
str(df_fetal_health)
summary(df_fetal_health)

################################################################################
######################## 1. Introduction #######################################
################################################################################

# Short introduction of the subject 

################################################################################
################# 2. Data Overview #############################################
################################################################################
library(ggplot2)
library(GGally)

df_fetal_health[, "fetal_health"] <- factor(df_fetal_health[, "fetal_health"])

ggpairs(data = df_fetal_health, columns = 1:11,
        aes(color = fetal_health, alpha = 0.5)) +
  scale_fill_manual(values = c("yellow3", "orange3", "red3")) + 
  scale_color_manual(values = c("yellow3", "orange3", "red3"))


################################################################################
########################## 3. EDA  #############################################
################################################################################

################################################################################
###################### 3.1 Hearth Rate Baseline ################################
################################################################################

library(gridExtra)

hist_HearthRate <- ggplot(data = df_fetal_health, 
                          aes(x = baseline.value, 
                              fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Density Baseline Value of Heart Beat") +
  ylab("Density") +
  xlab("Baseline Value")

box_HearthRate <- ggplot(data = df_fetal_health, 
                         aes(x = fetal_health,
                             y = baseline.value, 
                             fill = fetal_health)) +
  geom_boxplot(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Boxplot Baseline Value of Heart Beat") +
  ylab("Baseline Value") +
  xlab("Fetal Health Condition")


p <- ggplot(data = df_fetal_health, 
       aes(x = baseline.value, 
           fill = fetal_health)) +
  geom_histogram(alpha = 0.3)  + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ylab("Baseline Value") +
  xlab("Fetal Health Condition")
library(plotly)
ggplotly(p)
grid.arrange(hist_HearthRate, box_HearthRate, ncol = 2, nrow = 1)

################################################################################
#################### 3.2 Accelerations and Decelerations #######################
################################################################################
# boxplot format seems to be useless, we are using density in this case. We keep the code for the R shiny

# box_Acceleration <- ggplot(data = df_fetal_health, 
#                          aes(x = fetal_health,
#                              y = accelerations, 
#                              fill = fetal_health)) +
#   geom_boxplot() + 
#   scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
#   theme(
#     legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
#     legend.title = element_text(size = 10),
#     plot.title = element_text(color = "black", size = 20)
#   ) +
#   ggtitle("Accelerations") +
#   ylab("Acceleration per second") +
#   xlab("Fetal Health Condition")
# 
# box_Ldeceleration <- ggplot(data = df_fetal_health, 
#                            aes(x = fetal_health,
#                                y = light_decelerations, 
#                                fill = fetal_health)) +
#   geom_boxplot() + 
#   scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
#   theme(
#     legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
#     legend.title = element_text(size = 10),
#     plot.title = element_text(color = "black", size = 20)
#   ) +
#   ggtitle("Light Decelerations") +
#   ylab("Light decelerations per second") +
#   xlab("Fetal Health Condition")
# 
# box_Sdeceleration <- ggplot(data = df_fetal_health, 
#                             aes(x = fetal_health,
#                                 y = severe_decelerations, 
#                                 fill = fetal_health)) +
#   geom_boxplot() + 
#   scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
#   theme(
#     legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
#     legend.title = element_text(size = 10),
#     plot.title = element_text(color = "black", size = 20)
#   ) +
#   ggtitle("Severe Decelerations") +
#   ylab("Severe decelerations per second") +
#   xlab("Fetal Health Condition")
# 
# box_Pdeceleration <- ggplot(data = df_fetal_health, 
#                             aes(x = fetal_health,
#                                 y = prolongued_decelerations, 
#                                 fill = fetal_health)) +
#   geom_boxplot() + 
#   scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
#   theme(
#     legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
#     legend.title = element_text(size = 10),
#     plot.title = element_text(color = "black", size = 20)
#   ) +
#   ggtitle("Prolongued Decelerations") +
#   ylab("Prolongued decelerations per second") +
#   xlab("Fetal Health Condition")
# 
# grid.arrange(box_Acceleration,
#              box_Ldeceleration,
#              box_Sdeceleration,
#              box_Pdeceleration, 
#              ncol = 2, 
#              nrow = 2)

density_acceleration <- ggplot(data = df_fetal_health, 
                          aes(x = accelerations, 
                              fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Accelerations") +
  ylab("Density") +
  xlab("Accelerations per second")

density_light_decelerations <- ggplot(data = df_fetal_health, 
                          aes(x = light_decelerations, 
                              fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Light Decelerations (LD)") +
  ylab("Density") +
  xlab("Light Decelerations per second")

density_severe_decelerations <- ggplot(data = df_fetal_health, 
                          aes(x = severe_decelerations, 
                              fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Severe Decelerations (SD)") +
  ylab("Density") +
  xlab("Severe Decelerations per second")

density_prolongued_decelarations <- ggplot(data = df_fetal_health, 
                          aes(x = prolongued_decelerations, 
                              fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Prolongued Deceelerations (PD)") +
  ylab("Density") +
  xlab("Prolongued decelerations per second")

grid.arrange(density_acceleration,
             density_light_decelerations,
             density_severe_decelerations,
             density_prolongued_decelarations,
             ncol = 2, nrow = 2)

################################################################################
#################### 3.3 Movements #############################################
################################################################################
# boxplots are not interesting in this case, we switch to density

# box_fetalMovements <- ggplot(data = df_fetal_health, 
#                             aes(x = fetal_health,
#                                 y = uterine_contractions, 
#                                 fill = fetal_health)) +
#   geom_boxplot() + 
#   scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
#   theme(
#     legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
#     legend.title = element_text(size = 10),
#     plot.title = element_text(color = "black", size = 20)
#   ) +
#   ggtitle("Uterine Contractions") +
#   ylab("Uterine Contractions per second") +
#   xlab("Fetal Health Condition")
# 
# box_uterineMovements <- ggplot(data = df_fetal_health, 
#                                aes(x = fetal_health,
#                                    y = fetal_movement, 
#                                    fill = fetal_health)) +
#   geom_boxplot() + 
#   scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
#   theme(
#     legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
#     legend.title = element_text(size = 10),
#     plot.title = element_text(color = "black", size = 20)
#   ) +
#   ggtitle("Fetal Movements") +
#   ylab("Fetal movements per second") +
#   xlab("Fetal Health Condition")
# 
# grid.arrange(box_uterineMovements, 
#              box_fetalMovements,
#              ncol = 2, 
#              nrow = 1)

density_uterine_contractions <- ggplot(data = df_fetal_health, 
                          aes(x = uterine_contractions, 
                              fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Uterine Contractions") +
  ylab("Density") +
  xlab("Uterine Contractions per second")

density_fetal_movements <- ggplot(data = df_fetal_health, 
                          aes(x = fetal_movement, 
                              fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Foetal Movement") +
  ylab("Density") +
  xlab("Foetal Movements per second")

grid.arrange(density_uterine_contractions, density_fetal_movements, 
             nrow = 1, ncol = 2)

# is it interesting? seems the same as the boxplot

################################################################################
########################## 3.4 STV Vs LTV ######################################
################################################################################

density_STV <- ggplot(data = df_fetal_health, 
                                       aes(x = abnormal_short_term_variability, 
                                           fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Percentage of time with abnormal STV") +
  ylab("Density") +
  xlab("Percentage")

density_LTV <- ggplot(data = df_fetal_health, 
                                  aes(x = percentage_of_time_with_abnormal_long_term_variability, 
                                      fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Percentage of time with abnormal LTV") +
  ylab("Density") +
  xlab("Percentage")

grid.arrange(density_STV, density_LTV, 
             nrow = 2, ncol = 1)

################################################################################
########################## 3.5 Histogram ####################################### 
################################################################################

density_histogram_width <- ggplot(data = df_fetal_health, 
                      aes(x = histogram_width, 
                          fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Heart Rate Histogram Width (max HR - min HR)") +
  ylab("Density") +
  xlab("Heart Rate Width")

density_histogram_variance <- ggplot(data = df_fetal_health, 
                      aes(x = histogram_variance, 
                          fill = fetal_health)) +
  geom_density(alpha = 0.3) + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Hearth Rate Histogram Variance") +
  ylab("Density") +
  xlab("Variance")

grid.arrange(density_histogram_width, density_histogram_variance, 
             nrow = 2, ncol = 1)

################################################################################
########################## 3.6 Foetal Health ###################################
################################################################################

histogram_foetal_health <- ggplot(data = df_fetal_health, 
                                     aes(x = fetal_health, fill = fetal_health)) +
  geom_bar(stat = "count") + 
  scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
  theme(
    legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
    legend.title = element_text(size = 10),
    plot.title = element_text(color = "black", size = 20)
  ) +
  ggtitle("Foetal Health") +
  ylab("Frequency") +
  xlab("Fetal Health")
histogram_foetal_health

################################################################################
######################## 4. Machine Learning Models ############################
################################################################################

################################################################################
######################## 4.1 Creation of train/test data #######################
################################################################################

library(mlbench)
library(kernlab)
library(e1071)
library(caret)
library(dplyr)

set.seed(123)
indices <- createDataPartition(df_fetal_health$fetal_health, p = .75, list = F)
train <- df_fetal_health %>% slice(indices)

test_in <- df_fetal_health %>% slice(-indices) %>% select(-fetal_health)
test_in
test_truth <- df_fetal_health %>% slice(-indices) %>% pull(fetal_health)



################################################################################
########################### 4.1.1 SVM model ####################################
################################################################################

tune_out_linear <- tune(svm, fetal_health~., data = train, kernel = "linear",
                        ranges = list(cost = c(0.1,1,10,30)))
tune_out_linear$best.model

tune_out_radial <- tune(svm, fetal_health~., data = train, kernel = "radial",
                        ranges = list(cost = c(0.1,1,10),
                                      gamma = c(0.5,1,2,3,4)))
tune_out_radial$best.model
tune_out_radial$best.model$gamma

svm_linear <- svm(fetal_health~ ., train, kernel = "linear", 
                              scale = TRUE, cost = 0.1)

test_pred_svm_linear <- predict(svm_linear, test_in)
table(test_pred_svm_linear)

confusion_matrix_svm_linear <- confusionMatrix(test_pred_svm_linear, test_truth, mode = "everything")
as.numeric(confusion_matrix_svm_linear$byClass[, "Precision"])
as.numeric(confusion_matrix_svm_linear$byClass[1, "Balanced Accuracy"])
parameters <- c("Accuracy", "Precision", "Recall", "F1")
Class1 <- c(as.numeric(confusion_matrix_svm_linear$byClass[1, "Balanced Accuracy"]),
            as.numeric(confusion_matrix_svm_linear$byClass[1, "Precision"]),
            as.numeric(confusion_matrix_svm_linear$byClass[1, "Recall"]),
            as.numeric(confusion_matrix_svm_linear$byClass[1, "F1"]))
Class2 <- c(as.numeric(confusion_matrix_svm_linear$byClass[2, "Balanced Accuracy"]),
            as.numeric(confusion_matrix_svm_linear$byClass[2, "Precision"]),
            as.numeric(confusion_matrix_svm_linear$byClass[2, "Recall"]),
            as.numeric(confusion_matrix_svm_linear$byClass[2, "F1"]))
Class3 <- c(as.numeric(confusion_matrix_svm_linear$byClass[3, "Balanced Accuracy"]),
            as.numeric(confusion_matrix_svm_linear$byClass[3, "Precision"]),
            as.numeric(confusion_matrix_svm_linear$byClass[3, "Recall"]),
            as.numeric(confusion_matrix_svm_linear$byClass[3, "F1"]))
Mean <- c(mean(as.numeric(confusion_matrix_svm_linear$byClass[, "Balanced Accuracy"])),
          mean(as.numeric(confusion_matrix_svm_linear$byClass[, "Precision"])),
          mean(as.numeric(confusion_matrix_svm_linear$byClass[, "Recall"])),
          mean(as.numeric(confusion_matrix_svm_linear$byClass[, "F1"])))
df <- data.frame(paramaeters = parameters,
                 class1 = Class1,
                 class2 = Class2,
                 class3 = Class3,
                 mean_val = Mean)
confusion_matrix_svm_linear
# name <- c()
# importance <- c()
# for (i in 1:(length(train)-1)){
#   temp_dataset <- train[-c(i)]
#   temp_model <- svm(fetal_health~ ., temp_dataset, kernel = "linear", 
#                     scale = TRUE, cost = 0.1)
#   test_temp_model_one <- predict(temp_model, test_in)
#   confusion_matrix_temp_model_one <- confusionMatrix(test_temp_model_one, test_truth, mode = "everything")
#   importance[i] <- (1 - mean(confusion_matrix_temp_model_one$byClass[,"F1"])/
#                       mean(confusion_matrix_svm_linear$byClass[,"F1"])) * 100
#   name[i] <- colnames(df_fetal_health)[i]
# }
# features <- data.frame(names = name,
#                        importance = importance)
# library(ggplot2)
# library(plotly)
# p <- ggplot(data = features, aes(x = names, y = importance)) +
#   geom_histogram(stat = "identity", fill = heat.colors(21))
# ggplotly(p)
# 
# svm_linear <- svm(fetal_health~ abnormal_short_term_variability + 
#                     accelerations + 
#                     fetal_movement +
#                     histogram_median + 
#                     percentage_of_time_with_abnormal_long_term_variability,
#                   train, kernel = "linear", 
#                   scale = TRUE, cost = 0.1)
# 
# test_pred_svm_linear <- predict(svm_linear, test_in)
# table(test_pred_svm_linear)
# 
# confusion_matrix_svm_linear <- confusionMatrix(test_pred_svm_linear, test_truth, mode = "everything")
# mean(confusion_matrix_svm_linear$byClass[,"F1"])
# confusion_matrix_svm_linear

# quite good in predicting the first case. Less accurate in the second case. Quite good also for the third
# group. 

svm_radial <- svm(fetal_health~ ., train, kernel = "radial", 
                  scale = TRUE, cost = 10, gamma = 0.5)

test_pred_svm_radial <- predict(svm_radial, test_in)
table(test_pred_svm_radial)

confusion_matrix_svm_radial <- confusionMatrix(test_pred_svm_radial, test_truth, mode = "everything")
confusion_matrix_svm_radial

################################################################################
################## 4.1.2 Random Forest Classifier ##############################
################################################################################

library(randomForest)

random_forest <- randomForest(fetal_health~ ., train, proximity = TRUE)
random_forest

test_rfmodel <- predict(random_forest, test_in)
table(test_rfmodel)

confusion_matrix_rfmodel <- confusionMatrix(test_rfmodel, test_truth, mode = "everything")
confusion_matrix_rfmodel

################################################################################
################## 4.1.3 Multinomial Regression Model ##########################
################################################################################

library(nnet)

multinomial_model <- multinom(fetal_health~ ., train)

# summary(multinomial_model)
# 
# exp(coef(multinomial_model))

head(round(fitted(multinomial_model), 2))

test_multinomial_model <- predict(multinomial_model, newdata = test_in)

confusion_matrix_multinomial_model <- confusionMatrix(test_multinomial_model, test_truth, mode = "everything")
confusion_matrix_multinomial_model

################################################################################
##################### 4.1.4 Feature selection ##################################
################################################################################

# eliminating the comments, it is possible to try to boost the model through feature selection. 
# However, we believe that it was not worthy, since the result are not significant.

# train_predictors = train[ , -22]
# train_outcome = train[ , 22]
# 
# control_rfe = rfeControl(functions = rfFuncs, # random forest
#                          method = "repeatedcv", # repeated cv
#                          repeats = 5, # number of repeats
#                          number = 10) # number of folds
# set.seed(50)
# # Performing RFE
# result_rfe = rfe(x = train_predictors,
#                  y = train_outcome,
#                  sizes = c(1:21),
#                  rfeControl = control_rfe)
# 
# # summarising the results
# result_rfe$variables
# df_rfe <- df_fetal_health[, result_rfe$optVariables]
################################################################################
##################### 4.1.5 Testing Feature selection ##########################
################################################################################
# train_rfe <- train[, c(result_rfe$optVariables,"fetal_health")]
# svm_linear_opt <- svm(fetal_health~ ., train_rfe, kernel = "linear", 
#                   scale = TRUE, cost = 0.1)
# 
# test_pred_svm_linear_opt <- predict(svm_linear_opt, test_in)
# table(test_pred_svm_linear_opt)
# 
# confusion_matrix_svm_linear_opt <- confusionMatrix(test_pred_svm_linear_opt, test_truth, mode = "everything")
# confusion_matrix_svm_linear_opt
# confusion_matrix_svm_linear
# # Worst than before
# 
# svm_radial_opt <- svm(fetal_health~ ., train_rfe, kernel = "radial", 
#                   scale = TRUE, cost = 10, gamma = 0.5)
# 
# test_pred_svm_radial_opt <- predict(svm_radial_opt, test_in)
# 
# confusion_matrix_svm_radial_opt <- confusionMatrix(test_pred_svm_radial_opt, test_truth, mode = "everything")
# confusion_matrix_svm_radial_opt
# confusion_matrix_svm_radial
# # way better this model, reach the multinomial level
# 
# random_forest_opt <- randomForest(fetal_health~ ., 
#                                   train_rfe, proximity = TRUE)
# random_forest_opt
# 
# test_rfmodel_opt <- predict(random_forest_opt, test_in)
# table(test_rfmodel_opt)
# 
# confusion_matrix_rfmodel_opt <- confusionMatrix(test_rfmodel_opt, test_truth, mode = "everything")
# confusion_matrix_rfmodel_opt
# confusion_matrix_rfmodel
# # Better than the original model (not significantly)
# 
# multinomial_model_opt <- multinom(fetal_health~ ., train_rfe)
# 
# test_multinomial_model_opt <- predict(multinomial_model_opt, newdata = test_in)
# 
# confusion_matrix_multinomial_model_opt <- confusionMatrix(test_multinomial_model_opt, test_truth, mode = "everything")
# confusion_matrix_multinomial_model_opt
# confusion_matrix_multinomial_model
# # way worst

