# Diabetic Progression Prediction

- This project uses **linear regression** to predict the progression of diabetes in patients based on diagnostic measurements. The dataset contains features such as age, BMI, blood pressure, and other diagnostic metrics, which are analyzed to build a regression model that can predict future diabetic progression.

## Project Structure

```plaintext
Diabetic-Progression-Prediction/
├── data/                               # Dataset files
├── src/
│   ├── data_pruning.py                 # Data cleaning and preparation
│   ├── model_training.py               # Model training
│   └── model_evaluation.py             # Helper functions
├── README.md                           # Project overview and instructions
└── output/                             # Model results, metrics, and visualizations
```

## Dataset Features

The dataset used in this project includes diagnostic information related to diabetes progression. The key features are:

| Feature       | Description                                              |
|---------------|----------------------------------------------------------|
| **Age**       | Age of the patient                                       |
| **BMI**       | Body Mass Index, a measure of body fat                   |
| **BP**        | Average blood pressure                                   |
| **S1-S6**     | Serum measurements and other metabolic indicators        |
| **Y**         | Target variable indicating disease progression after 1 year |

## Getting Started

To get started, follow these steps to clone the repository and install the required dependencies.

### Clone the Repository

```bash
git clone https://github.com/Naoldaba/Diabetic-Progression-Prediction.git
cd Diabetic-Progression-Prediction
```

## Comprehensive Report 
https://docs.google.com/document/d/1lPWSmDehQ5Kg7kRnBjG9yeYR-j7fPJw4Vds-UXjOwSs/edit?usp=sharing