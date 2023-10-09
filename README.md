# Foetal Health Prediction Project

## Project Folders

### `foetal_health_project`

This folder contains the core project files:

- `Data`: This directory holds the dataset used in the project.
- `FoetalHealthPrediction`: This directory contains the R scripts used for analysis and the Shiny app.
- `PresentationPP`: This directory includes the preliminary study and the final presentation documents.
- `foetalhealthCTG.html`: A complete report of the project in HTML format.
- `foetalhealthCTG.Rmd`: An Rmarkdown file containing the script used in the project.
- `readMe.txt`: The file you are currently reading.

### `Data`

This folder contains the project dataset in the file `fetal_health.csv`.

### `FoetalHealthPrediction`

- `project_script_DSinH.R`: An R script used in the Shiny app and the Rmarkdown file for analysis.
- `app.R`: A Shiny app that dynamically displays project results from the Rmarkdown file.

### `PresentationPP`

This folder contains project presentation documents:

- `Preliminary_Study.pdf`: A preliminary study outlining the project's initial concept and methodology.
- `Foetal_Health.pdf`: A PDF file for the final project presentation.

## Suggestions

1. **Adjust File Paths**: If you intend to run the code locally, make sure to adjust the file paths. Here are the lines of code that need modification:
    - In `foetalhealthCTG.Rmd`: 
        - Line 67: `df_fetal_health <- read.csv("YOUR PATH/Data/fetal_health.csv")` or (if you are in this file directory) `read.csv("Data/fetal_health.csv")`
    - In `FoetalHealthPrediction/app.R`:
        - Line 13: `source("YOUR PATH/FoetalHealthPrediction/project_script_DSinH.R")` or (if you are in this file directory) `source("project_script_DSinH.R")`
    - In `FoetalHealthPrediction/project_script_DSinH.R`: 
        - Line 1: `df_fetal_health <- read.csv("YOUR PATH/Data/fetal_health.csv")` or (if you are in this file directory) `df_fetal_health <- read.csv("../Data/fetal_health.csv")`

2. **Install Required Libraries**: Make sure to install the following libraries to run the code successfully:
    - "ggplot2", "plotly", "GGally", "shiny", "shinydashboard", "dplyr", "tidyverse", "readr", "shinythemes", "caret", "kernlab", "mlbench", "e1071", "randomForest", "nnet"

## Importance of Cardiotocography (CTG) in Medicine

Cardiotocography (CTG) is a crucial technology in the medical sector, especially in obstetrics. It is used to monitor the heart rate of a fetus and the uterine contractions of a pregnant woman during pregnancy and labor. CTG plays a vital role in identifying potential issues with the baby's health, allowing doctors to take timely interventions when necessary. Its non-invasive nature makes it a valuable tool for ensuring the well-being of both the mother and the unborn child.

## Relevance of Project Methods for Medical Professionals

The methods analyzed in this project hold great potential for medical professionals. By leveraging data analysis and machine learning techniques, doctors can spot potential critical situations in real-time, enhancing their ability to provide optimal care during pregnancy and labor. The project's findings may contribute to the development of more accurate and reliable predictive models, further improving the healthcare sector's ability to ensure safe deliveries and healthy newborns.
