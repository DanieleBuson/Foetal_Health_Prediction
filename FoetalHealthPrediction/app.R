library(shiny)
library(ggplot2)
library(shinydashboard)
library(plotly)
library(nnet)
library(randomForest)
library(caret)
library(e1071)
library(caret)
library(dplyr)

# use it just once
source("project_script_DSinH.R")

ui <- dashboardPage(
  skin = "green",
  dashboardHeader(title = "Foetal Health using CTG data",
                  titleWidth = 350),
  dashboardSidebar(
    sidebarMenu(
      menuItem(" Introduction", tabName = "intro", icon = icon("stethoscope")),
      menuItem(" EDA", tabName = "eda", icon = icon("chart-simple")),
      menuItem(" Models", tabName = "mod", icon = icon("network-wired")),
      menuItem(" About us", tabName = "about", icon = icon("linkedin")),
      menuItem(" Sources", tabName = "sources", icon = icon("newspaper"))
    )
  ),
  dashboardBody(
    
    tabItems(
      tabItem("intro",
              fluidRow(
                box( width = 12,
                column(
                  width = 4,
                  img(src = "https://img.24baby.nl/2017/11/Foetus.jpg",
                      align = "center", style = paste0("width: 100%; height: ", "30em", ";"))
                ),
                column(
                  width = 8,
                  titlePanel(h1("Introduction")),
                      div(h4("Analysis, Classification and Prediction of foetal health through supervised learning models."),
                          p("Reduction of child mortality is reflected in several of the United Nations’ 
                          Sustainable Development Goals and is a key indicator of human progress. In the 
                          last three decades, the world made remarkable progress in child survival. 
                          In the ’90s, 1 in 11 children died before the age of five. Nowadays, the 
                          ratio is 1 in 26 children. (1) The UN expects to lower this percentage to 25
                          deaths in 1000 births by 2030. Another aspect, alongside this, that is crucial 
                          in the Sustainable Development Goals is maternal mortality. About 287000 women 
                          died during and following pregnancy and childbirth in 2020. Almost 95% of all 
                          maternal deaths occurred in low and lower-middle-income countries in 2020, and 
                          most could have been prevented. (2) 
                          In this scenario, we conducted our study in order to understand if it was feasible 
                          to create supervised learning models through CTG data. 
                          Moreover, we would like to know whether the model can be used by 
                          doctors and obstetricians in their daily tasks. 
                          Finally, we find it necessary to specify that we 
                          do not want our model to substitute in any way professional figures. 
                          On the contrary, we aim to create a tool that can assist them in
                           double-checking their diagnosis.")
                          )
              )
              )),
              fluidRow(
                box(width = 12,
                column(h4("About CTG"),
                    p("Cardiotocography (CTGs) is a simple and cost-accessible option 
                    to assess foetal health, allowing healthcare professionals to take 
                    action in order to prevent child and maternal mortality. The equipment 
                    itself works by sending ultrasound pulses and reading their response, 
                    thus shedding light on foetal heart rate (FHR), foetal movements, 
                    uterine contractions and more.
                    More than 350 years ago, doctors were already studying foetal 
                    heart sounds. The first tool used for this purpose was the Pinard 
                    horn, an ago mechanical stethoscope, created approximately 200 
                    years ago. 
                    Modern-day CTG was developed and introduced in the '50s, while the 
                    first commercial foetal monitor was released in 1968.
                    CTG monitoring is widely used during labour to assess foetal well-being by identifying
                    babies at risk of hypoxia (lack of oxygen). 
                    Our study is founded on CTG data, as well as our models. 
                    The main reason is that we found CTG a very democratic technique 
                    since the costs to perform it are limited. As a consequence, we believe 
                    that creating a suitable model could assist doctors and obstetricians in 
                    helping people from all over the world and of all social classes. 
                    You can find all the models in the dedicated section (Models)."),
                    width = 8
                ),
                column(img(src = "https://cdn.cdnparenting.com/articles/2018/01/612984125-H.jpg",
                        align = "center", style = paste0("width: 100%; height: ", "30em", ";")),
                    width = 4
                )
              ))
      ),
      tabItem("eda",
              fluidRow(box(width = 12,
                column(width = 2),
                column(titlePanel(h1("Exploratory Data Analysis", align = "center")),
                    selectInput("eda_plot", "Feature: ", 
                                c("Baseline Value" = "baseline.value", 
                                  "Accelerations" = "accelerations", 
                                  "Fetal Movement" = "fetal_movement",
                                  "Uterine Contractions" = "uterine_contractions",
                                  "Light Decelerations" = "light_decelerations",
                                  "Severe Decelerations" = "severe_decelerations",
                                  "Prolongued Decelerations" = "prolongued_decelerations",
                                  "Abnormal percentage of time with STV" = "abnormal_short_term_variability",
                                  "Mean Value of STV" = "mean_value_of_short_term_variability", 
                                  "Abnormal percentage of time with LTV" = "percentage_of_time_with_abnormal_long_term_variability",
                                  "Mean Value of LTV" = "mean_value_of_long_term_variability", 
                                  "CTG Histogram Width" = "histogram_width",
                                  "Minimum of CTG Histogram" = "histogram_min",
                                  "Maximum of CTG Histogram" = "histogram_max", 
                                  "Number of Peaks of CTG Histogram" = "histogram_number_of_peaks",
                                  "Number of Zeroes in the CTG Histogram" = "histogram_number_of_zeroes",
                                  "CTG Histogram Mode" = "histogram_mode",
                                  "CTG Histogram Mean" = "histogram_mean",
                                  "CTG Histogram Median" = "histogram_median",
                                  "CTG Histogram Variance" = "histogram_variance",
                                  "CTG Histogram Tendency" = "histogram_tendency")),
                    width = 8),
                column(width = 2)
              )),
              fluidRow(
                box(
                  width = 12,
                  box(width = 4,
                      h4("Density Plot", align = "center"),
                      plotlyOutput("density"), background = "green"),
                  box(width = 4,
                      h4("Histogram", align = "center"),
                      plotlyOutput("histogram"), background = "green"),
                  box(width = 4,
                      h4("Boxplot", align = "center"),
                      plotlyOutput("boxplot"), background = "green")
                )
              ),
              fluidRow(
                box(width = 12,
                    box(titlePanel(h1("Analysis of Foetal Health", align = "center")),
                           p("On the side, the graph shows the distribution across the 
                           three levels (normal, suspect and pathological). The normal class seems to be 
                           the most populated, with more than 1600 cases. The other categories 
                             (suspect and pathological) have altogether around 500 records. 
                             This situation was expected. However, it could cause some technical
                             issues in the classification models."),
                           width = 4),
                    box(width = 8,
                           h4("Histogram", align = "center"),
                           plotlyOutput("histogramfh"), background = "green")
                    )
              )
      ),
      tabItem("mod",
              fluidRow(
                box(width = 12, 
                    column(width = 2),
                    column(titlePanel(h1("Supervised Learning Models", align = "center")),
                           selectInput("model_sel", "Models: ", 
                                       c("Linear Kernel SVM",
                                         "Radial Kernel SVM", 
                                         "Random Forest Classifier",
                                         "Multinomial Logistic Regression")),
                           width = 8),
                    column(width = 2)),
                box(width = 12, 
                    h4("Model Performances", align = "center"),
                    p("Below you can see the preformances of the selected supervised lerning method.", align = "center")),
                    column(2),
                    column(div(tableOutput("outcome"), align = "center"), width = 8),
                    column(2)
              ),
              fluidRow(
                box(width = 12, 
                    column(width = 2),
                    column(titlePanel(h4("Try out the model!", align = "center")),
                           selectInput("model_sel2", "Select the model: ", 
                                       c("Linear Kernel SVM",
                                         "Radial Kernel SVM", 
                                         "Random Forest Classifier",
                                         "Multinomial Logistic Regression")),
                           width = 8),
                    column(width = 2)),
                box(width = 4,
                    numericInput("baseline", "Baseline values: ", min = 100, max = 180, step = 1, value = 133),
                    numericInput("uterine", "Uterine contractions per second: ", min = 0, max = 0.025, step = 0.001, value = 0.004),
                    numericInput("prolongued", "Prolongued decelerations per second: ", min = 0, max = 0.006, step = 0.001, value = 0),
                    numericInput("percentageLTV", "Percentage of time with abnormal LTV: ", min = 0, max = 100, step = 1, value = 0),
                    numericInput("minimum", "Histogram minimum: ", min = 50, max = 165, step = 1, value = 93),
                    numericInput("zeroes", "Histogram number of zeros: ", min = 0, max = 15, step = 1, value = 0),
                    numericInput("median", "Histogram median: ", min = 50, max = 200, step = 1, value = 139)),
                box(width = 4,
                    numericInput("accelerations", "Accelerations per second: ", min = 0, max = 0.025, step = 0.001, value = 0.002),
                    numericInput("light", "Light decelerations per second: ", min = 0, max = 0.025, step = 0.001, value = 0),
                    numericInput("percentageSTV", "Percentage of time with abnormal STV: ", min = 0, max = 100, step = 1, value = 49),
                    numericInput("meanLTV", "Mean value of LTV: ", min = 0, max = 60, step = 0.1, value = 7.5),
                    numericInput("maximum", "Histogram maximum: ", min = 100, max = 250, step = 1, value = 162),
                    numericInput("mode", "Histogram mode: ", min = 50, max = 200, step = 1, value = 139),
                    numericInput("variance", "Histogram variance: ", min = 0, max = 300, step = 1, value = 7)),
                box(width = 4,
                    numericInput("movement", "Foetal Movements per second: ", min = 0, max = 0.65, step = 0.01, value = 0),
                    numericInput("severe", "Severe decelerations per second: ", min = 0, max = 0.002, step = 0.001, value = 0),
                    numericInput("meanSTV", "Mean value of STV: ", min = 0, max = 7, step = 0.1, value = 1.2),
                    numericInput("width", "Histogram width: ", min = 0, max = 190, step = 1, value = 67),
                    numericInput("peaks", "Histogram number of peaks: ", min = 0, max = 20, step = 1, value = 3),
                    numericInput("mean", "Histogram mean: ", min = 50, max = 200, step = 1, value = 136),
                    numericInput("tendency", "Histogram tendency: ", min = -1, max = 1, step = 1, value = 0))
              ),
              fluidRow(
                width = 12, 
                column(width = 2),
                column(width = 8,
                       div(actionButton("button", label = "Test the Model"), align = "center"),
                       div(tableOutput("result"), align = "center")),
                column(width = 2)
              )
      ),
      tabItem("about",
              fluidRow(column(h1("About Us", align = "center"), width = 12)),
              fluidRow(box(img(src = "https://media.licdn.com/dms/image/C4D03AQHHLDNTI0nDXg/profile-displayphoto-shrink_800_800/0/1594767653272?e=1686182400&v=beta&t=h0ItNF27P86kBDmY6b4OVBZ5XkBidCeCA8UFvba-2b4",
                                   align = "center", style = paste0("width: 100%; height: ", "30em", ";")),
                           h4("Jonas Renfer", align = "center"),
                           h4("AI Project Manager at Helvetia and Student MSc Applied Information & Data Science at HSLU"),
                           h6("https://www.linkedin.com/in/jonasrenfer/"),
                           width = 4, 
                           background = "green"),
                       box(img(src = "https://media.licdn.com/dms/image/C5603AQFeCj5KoVhisA/profile-displayphoto-shrink_200_200/0/1520030087896?e=1686182400&v=beta&t=oQCrelIsPHLNf8M7qoaQyg2SF2BAOS5KnZOb-Dnqb3E",
                               align = "center", style = paste0("width: 100%; height: ", "30em", ";")),
                           h4("Saša Ljubisavljević"),
                           h4("Support Performance Management and Student MSc Applied Information & Data Science at HSLU"),
                           h6("https://www.linkedin.com/in/sa%C5%A1a-ljubisavljevi%C4%87-77a427155/"),
                           background = "green",
                           width = 4),
                       box(img(src = "https://media.licdn.com/dms/image/C4D03AQHdsQZ43lJd0Q/profile-displayphoto-shrink_800_800/0/1619939966754?e=1686182400&v=beta&t=iIVlj2x-Li5MUFOiDqbNHi5n7GztWTvZIQ2bkSub_38",
                               align = "center", style = paste0("width: 100%; height: ", "30em", ";")),
                           h4("Daniele Buson"),
                           h4("Student MSc Applied Information & Data Science at HSLU"),
                           h6("https://www.linkedin.com/in/daniele-buson-325471168/"),
                           h6("https://github.com/DanieleBuson"),
                           background = "green",
                           width = 4),
                      )
              ),
      tabItem("sources",
              fluidRow(
                column(h1("Sources and References", align = "center"), width = 12),
                h4("Here is the list of sources and references that we used in our study", align = "center")
              ),
              fluidRow(
                column(2),
                box(
                  h4("https://data.unicef.org/topic/child-survival/under-five-mortality/", align = "center"),
                  h4("https://www.who.int/news-room/fact-sheets/detail/maternal-mortality", align = "center"),
                  h4("https://www.ncbi.nlm.nih.gov/books/NBK557393/", align = "center"),
                  h4("https://pubmed.ncbi.nlm.nih.gov/11874624/", align = "center"),
                  h4("https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification", align = "center"),
                  h4("https://www.sciencedirect.com/topics/nursing-and-health-professions/fetus-heart-rate", align = "center"),
                  h4("https://www.aafp.org/pubs/afp/issues/1999/0501/p2487.html", align = "center"),
                  h4("https://www.ncbi.nlm.nih.gov/books/NBK532927/", align = "center"),
                  h4("https://www.ncbi.nlm.nih.gov/books/NBK470566/", align = "center"),
                  h4("https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-016-0845-8", align = "center"),
                  h4("https://link.springer.com/chapter/10.1007/978-3-642-70358-4_20", align = "center"),
                  h4("https://www.medtronicacademy.com/features/rate-histograms-feature", align = "center"),
                  h4("https://en.wikipedia.org/wiki/Support_vector_machine", align = "center"),
                  h4("https://en.wikipedia.org/wiki/Random_forest", align = "center"), width = 8),
                column(2)
              ),
              fluidRow(
                column(h3("Thanks for the attention!"), align = "center", width = 12)
              )
              )
)))

server <- function(input, output) {
  
  ##################eda##################

  output$density <- renderPlotly({
    first <- df_fetal_health[, input$eda_plot]
    second <- df_fetal_health$fetal_health
    temp_df <- data.frame(first = first,
                          second = second)
    print(input$eda_plot)
    p <- ggplot(data = temp_df, 
                aes(x = first, 
                    fill = second)) +
      geom_density(alpha = 0.3) + 
      scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3"))  +
      theme(
        legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
        legend.title = element_text(size = 10),
        plot.title = element_text(color = "black", size = 20)
      ) +
      ylab("Density") +
      xlab(input$eda_plot)
    ggplotly(p)
  })
  
  output$histogram <- renderPlotly({
    first <- df_fetal_health[, input$eda_plot]
    second <- df_fetal_health$fetal_health
    temp_df <- data.frame(first = first,
                          second = second)
    p <- ggplot(data = temp_df, 
                aes(x = first, 
                    fill = second)) +
      geom_histogram(alpha = 0.3)  + 
      scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
      theme(
        legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
        legend.title = element_text(size = 10),
        plot.title = element_text(color = "black", size = 20)
      ) +
      ylab("Frequency") +
      xlab(input$eda_plot)
    ggplotly(p)
  })
  
  output$boxplot <- renderPlotly({
    first <- df_fetal_health[, input$eda_plot]
    second <- df_fetal_health$fetal_health
    temp_df <- data.frame(first = first,
                          second = second)
    p <- ggplot(data = temp_df,
                aes(x = second,
                    y = first,
                    fill = second)) +
      geom_boxplot(alpha = 0.3) +
      scale_fill_manual(name = "Fetal Health Condition", values = c("yellow3", "orange3", "red3")) +
      theme(
        legend.background = element_rect(fill = "white", color = "black", linetype = "solid"),
        legend.title = element_text(size = 10),
        plot.title = element_text(color = "black", size = 20)
      ) +
      ylab(input$eda_plot) +
      xlab("Fetal Health Condition")
    ggplotly(p)
  })
  
  output$histogramfh <- renderPlotly({
    p <-ggplot(data = df_fetal_health, aes(x = fetal_health, fill = fetal_health)) +
      geom_bar(stat = "count", alpha = 0.3) + 
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
    ggplotly(p)
  })
  
  ###############model###########################################################
  
  output$outcome <- renderTable({
    
    if (input$model_sel == "Linear Kernel SVM"){
      svm_linear <- svm(fetal_health~ ., train, kernel = "linear", 
                        scale = TRUE, cost = 0.1)
      test_pred_svm_linear <- predict(svm_linear, test_in)
      confusion_matrix_svm_linear <- confusionMatrix(test_pred_svm_linear, test_truth, mode = "everything")
      
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
      df <- data.frame(Parameter = parameters,
                       Class_1 = Class1,
                       Class_2 = Class2,
                       Class_3 = Class3,
                       Mean_Value = Mean)
    }
    if (input$model_sel == "Radial Kernel SVM"){
      svm_radial <- svm(fetal_health~ ., train, kernel = "radial", 
                        scale = TRUE, cost = 10, gamma = 0.5)
      test_pred_svm_radial <- predict(svm_radial, test_in)
      confusion_matrix_svm_radial <- confusionMatrix(test_pred_svm_radial, test_truth, mode = "everything")
      
      parameters <- c("Accuracy", "Precision", "Recall", "F1")
      Class1 <- c(as.numeric(confusion_matrix_svm_radial$byClass[1, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[1, "Precision"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[1, "Recall"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[1, "F1"]))
      Class2 <- c(as.numeric(confusion_matrix_svm_radial$byClass[2, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[2, "Precision"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[2, "Recall"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[2, "F1"]))
      Class3 <- c(as.numeric(confusion_matrix_svm_radial$byClass[3, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[3, "Precision"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[3, "Recall"]),
                  as.numeric(confusion_matrix_svm_radial$byClass[3, "F1"]))
      Mean <- c(mean(as.numeric(confusion_matrix_svm_radial$byClass[, "Balanced Accuracy"])),
                mean(as.numeric(confusion_matrix_svm_radial$byClass[, "Precision"])),
                mean(as.numeric(confusion_matrix_svm_radial$byClass[, "Recall"])),
                mean(as.numeric(confusion_matrix_svm_radial$byClass[, "F1"])))
      df <- data.frame(Parameter = parameters,
                       Class_1 = Class1,
                       Class_2 = Class2,
                       Class_3 = Class3,
                       Mean_Value = Mean)
    }
    if (input$model_sel == "Random Forest Classifier"){
      random_forest <- randomForest(fetal_health~ ., train, proximity = TRUE)
      test_rfmodel <- predict(random_forest, test_in)
      confusion_matrix_rfmodel <- confusionMatrix(test_rfmodel, test_truth, mode = "everything")
      
      parameters <- c("Accuracy", "Precision", "Recall", "F1")
      Class1 <- c(as.numeric(confusion_matrix_rfmodel$byClass[1, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[1, "Precision"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[1, "Recall"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[1, "F1"]))
      Class2 <- c(as.numeric(confusion_matrix_rfmodel$byClass[2, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[2, "Precision"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[2, "Recall"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[2, "F1"]))
      Class3 <- c(as.numeric(confusion_matrix_rfmodel$byClass[3, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[3, "Precision"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[3, "Recall"]),
                  as.numeric(confusion_matrix_rfmodel$byClass[3, "F1"]))
      Mean <- c(mean(as.numeric(confusion_matrix_rfmodel$byClass[, "Balanced Accuracy"])),
                mean(as.numeric(confusion_matrix_rfmodel$byClass[, "Precision"])),
                mean(as.numeric(confusion_matrix_rfmodel$byClass[, "Recall"])),
                mean(as.numeric(confusion_matrix_rfmodel$byClass[, "F1"])))
      df <- data.frame(Parameter = parameters,
                       Class_1 = Class1,
                       Class_2 = Class2,
                       Class_3 = Class3,
                       Mean_Value = Mean)
    }
    if (input$model_sel == "Multinomial Logistic Regression"){
      multinomial_model <- multinom(fetal_health~ ., train)
      test_multinomial_model <- predict(multinomial_model, newdata = test_in)
      confusion_matrix_multinomial_model <- confusionMatrix(test_multinomial_model, test_truth, mode = "everything")
      
      parameters <- c("Accuracy", "Precision", "Recall", "F1")
      Class1 <- c(as.numeric(confusion_matrix_multinomial_model$byClass[1, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[1, "Precision"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[1, "Recall"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[1, "F1"]))
      Class2 <- c(as.numeric(confusion_matrix_multinomial_model$byClass[2, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[2, "Precision"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[2, "Recall"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[2, "F1"]))
      Class3 <- c(as.numeric(confusion_matrix_multinomial_model$byClass[3, "Balanced Accuracy"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[3, "Precision"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[3, "Recall"]),
                  as.numeric(confusion_matrix_multinomial_model$byClass[3, "F1"]))
      Mean <- c(mean(as.numeric(confusion_matrix_multinomial_model$byClass[, "Balanced Accuracy"])),
                mean(as.numeric(confusion_matrix_multinomial_model$byClass[, "Precision"])),
                mean(as.numeric(confusion_matrix_multinomial_model$byClass[, "Recall"])),
                mean(as.numeric(confusion_matrix_multinomial_model$byClass[, "F1"])))
      df <- data.frame(Parameter = parameters,
                       Class_1 = Class1,
                       Class_2 = Class2,
                       Class_3 = Class3,
                       Mean_Value = Mean)
    }
    df
  })
  
  output$result <- renderTable({
    columns <- c("baseline.value",
                 "accelerations",
                 "fetal_movement",
                 "uterine_contractions",
                 "light_decelerations",
                 "severe_decelerations",
                 "prolongued_decelerations",
                 "abnormal_short_term_variability",
                 "mean_value_of_short_term_variability",
                 "percentage_of_time_with_abnormal_long_term_variability",
                 "mean_value_of_long_term_variability",
                 "histogram_width",
                 "histogram_min",
                 "histogram_max",
                 "histogram_number_of_peaks",
                 "histogram_number_of_zeroes",
                 "histogram_mode",
                 "histogram_mean",
                 "histogram_median",
                 "histogram_variance",
                 "histogram_tendency")
    df <- data.frame(matrix(nrow = 0, ncol = length(columns)))
    colnames(df) <- columns
    df[1,1] <- input$baseline
    df[1,2] <- input$accelerations
    df[1,3] <- input$movement
    df[1,4] <- input$uterine
    df[1,5] <- input$light
    df[1,6] <- input$severe
    df[1,7] <- input$prolongued
    df[1,8] <- input$percentageSTV
    df[1,9] <- input$meanSTV
    df[1,10] <- input$percentageLTV
    df[1,11] <- input$meanLTV
    df[1,12] <- input$width
    df[1,13] <- input$minimum
    df[1,14] <- input$maximum
    df[1,15] <- input$peaks
    df[1,16] <- input$zeroes
    df[1,17] <- input$mode
    df[1,18] <- input$mean
    df[1,19] <- input$median
    df[1,20] <- input$variance
    df[1,21] <- input$tendency
    
    
    cols <- c("Foetus Health Prediction") 
    result <- data.frame(matrix(nrow = 0, ncol = length(cols)))
    colnames(result) <- cols
    
    if (as.integer(input$button) > 0){

      if (input$model_sel == "Linear Kernel SVM"){
        
        test_pred_svm_linear <- predict(svm_linear, df)
        if (as.integer(test_pred_svm_linear[1]) == 1){
          result[1,1] = "Normal"
        }
        if (as.integer(test_pred_svm_linear[1]) == 2){
          result[1,1] = "Suspect"
        }
        if (as.integer(test_pred_svm_linear[1]) == 3){
          result[1,1] = "Pathological"
        }
      }

      if (input$model_sel == "Radial Kernel SVM"){
        
        test_pred_svm_radial <- predict(svm_radial, df)
        if (as.integer(test_pred_svm_radial[1]) == 1){
          result[1,1] = "Normal"
        }
        if (as.integer(test_pred_svm_radial[1]) == 2){
          result[1,1] = "Suspect"
        }
        if (as.integer(test_pred_svm_radial[1]) == 3){
          result[1,1] = "Pathological"
        }
      }

      if (input$model_sel == "Random Forest Classifier"){
        
        test_rfmodel <- predict(random_forest, df)
        if (as.integer(test_rfmodel[1]) == 1){
          result[1,1] = "Normal"
        }
        if (as.integer(test_rfmodel[1]) == 2){
          result[1,1] = "Suspect"
        }
        if (as.integer(test_rfmodel[1]) == 3){
          result[1,1] = "Pathological"
        }
      }

      if (input$model_sel == "Multinomial Logistic Regression"){
        
        test_multinomial_model <- predict(multinomial_model, newdata = df)
        if (as.integer(test_multinomial_model[1]) == 1){
          result[1,1] = "Normal"
        }
        if (as.integer(test_multinomial_model[1]) == 2){
          result[1,1] = "Suspect"
        }
        if (as.integer(test_multinomial_model[1]) == 3){
          result[1,1] = "Pathological"
        }
      }
    }
    result
  }, align = "c")
  
}

shinyApp(ui = ui, server = server)
