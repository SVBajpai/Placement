import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import joblib

# Reading the CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_dataset():
    print("****************************Set Dataset******************************")
    # Load dataset
    print(BASE_DIR, "BASE_DIR")
    data_csv = os.path.join(str(BASE_DIR), 'static', "Placement_Data_Full_Class.csv")
    df = pd.read_csv(data_csv)

    # Data Preprocessing
    # Dropping unnecessary columns
    df = df.drop(['sl_no', 'salary', 'mba_p', 'specialisation'], axis=1)

    # Handling missing values
    df = df.dropna()

    # Initialize LabelEncoders for categorical columns
    label_encoders = {}
    categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'status']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separating features and target
    X = df.drop('status', axis=1)
    y = df['status']

    # Scaling numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return [X_train, X_test, y_train, y_test], scaler, label_encoders, df, X, y

def train_dataset():
    print("****************************Training Dataset******************************")
    testing_data, scaler, label_encoders, df, X, y = set_dataset()
    X_train, X_test, y_train, y_test = testing_data

    # Training Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluating the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

    # Save the model, scaler, and label encoders
    joblib.dump(rf_model, os.path.join(BASE_DIR, 'static', 'models', 'rf_model.pkl'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'static', 'models', 'scaler.pkl'))
    for col, le in label_encoders.items():
        joblib.dump(le, os.path.join(BASE_DIR, 'static', 'models', f'le_{col}.pkl'))

    return label_encoders, scaler, rf_model

def load_models():
    # Load the model, scaler, and label encoders
    print("****************************LOADING MODELS******************************")
    rf_model = joblib.load(os.path.join(BASE_DIR, 'static', 'models', 'rf_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'static', 'models', 'scaler.pkl'))
    label_encoders = {}
    categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'status']
    
    for col in categorical_cols:
        label_encoders[col] = joblib.load(os.path.join(BASE_DIR, 'static', 'models', f'le_{col}.pkl'))
    
    return label_encoders, scaler, rf_model

# Function to predict placement for new raw input
def predict_placement(raw_input):
    """
    Predict placement status for a new raw input.
    Input should be a dictionary with keys matching the dataset columns (excluding sl_no, salary, status).
    Returns the predicted status ('Placed' or 'Not Placed') and probability.
    """
    print("****************************predict_placement******************************")
    
    # label_encoders, scaler, rf_model = train_dataset()
    label_encoders, scaler, rf_model = load_models()
    # Expected columns in order
    columns = ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p']

    # Create a DataFrame from the input
    input_df = pd.DataFrame([raw_input], columns=columns)

    # Encode categorical variables
    for col in ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex']:
        try:
            # Transform using the trained LabelEncoder
            input_df[col] = label_encoders[col].transform(input_df[col])
        except ValueError:
            raise ValueError(f"Invalid value for {col}. Must be one of {list(label_encoders[col].classes_)}")

    # Scale numerical features
    numerical_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p']
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][1]  # Probability of 'Placed'

    # Convert numerical prediction back to string
    predicted_status = label_encoders['status'].inverse_transform([prediction])[0]

    return predicted_status, probability

# Example usage of the prediction function
example_input = {
    'gender': 'M',
    'ssc_p': 75.0,
    'ssc_b': 'Central',
    'hsc_p': 70.0,
    'hsc_b': 'Others',
    'hsc_s': 'Science',
    'degree_p': 68.0,
    'degree_t': 'Sci&Tech',
    'workex': 'Yes',
    'etest_p': 80.0,
}

# try:
#     status, prob = predict_placement(example_input)
#     print(f"\nPrediction for example input:")
#     print(f"Placement Status: {status}")
#     print(f"Probability of Placement: {prob:.2%}")
# except ValueError as e:
#     print(f"Error: {e}")

# Visualizations (unchanged from previous code)
# Feature Importance Plot
def plot_feature_graph():
    testing_data, scaler, label_encoders, df, X, f = set_dataset()
    label_encoders, scaler, rf_model = load_models()
    plt.figure(figsize=(10, 6))
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    feature_importance.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Importance in Placement Prediction')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    fig_path = os.path.join(str(BASE_DIR), 'static', 'graph', "feature_importance.png")
    plt.savefig(fig_path)
    plt.close()
    msg = """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <h3>🖼️ About the Image: Feature Importance Graph</h3>
                <p>The image displays a bar chart titled <strong>"Feature Importance in Placement Prediction"</strong>.</p>

                <p>📊 Each bar represents a feature used in the prediction model (like CGPA, number of internships, communication skills, etc.).</p>

                <p>🏷️ <strong>X-axis:</strong> Lists the names of the input features.</p>

                <p>📏 <strong>Y-axis:</strong> Shows the importance score, indicating how much each feature contributes to the model's decision.</p>

                <p>🔼 Taller bars mean that feature plays a <strong>bigger role</strong> in predicting whether a student gets placed.</p>

                <p>📉 Shorter bars indicate <strong>less influence</strong> on the prediction outcome.</p>

                <p>This visualization helps highlight which student profile factors are most critical for placement success according to the trained model.</p>
            </div>
        """
    return fig_path, msg

# Distribution of Prediction Probabilities
def plot_probability():
    testing_data, scaler, label_encoders, df, X, y = set_dataset()
    label_encoders, scaler, rf_model = load_models()
    X_train, X_test, y_train, y_test = testing_data

    y_prob = rf_model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(10, 6))
    sns.histplot(y_prob, bins=20, kde=True)
    plt.title('Distribution of Placement Probability')
    plt.xlabel('Probability of Being Placed')
    plt.ylabel('Count')
    fig_path = os.path.join(str(BASE_DIR), 'static', 'graph', "probability_distribution.png")
    plt.savefig(fig_path)
    plt.close()
    msg = """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h3>🖼️ About the Image: Distribution of Placement Probability</h3>
        <p>The image displays a histogram titled <strong>"Distribution of Placement Probability"</strong>.</p>

        <p>📈 The chart shows how likely students are to be placed, based on the model's predictions.</p>

        <p>🔵 <strong>X-axis:</strong> Represents the predicted probability of getting placed (from 0 to 1).</p>

        <p>🔢 <strong>Y-axis:</strong> Indicates the number of students (count) that fall within each probability range.</p>

        <p>🌊 The smooth curve (if shown) represents the <strong>probability density estimate</strong>, helping to visualize the shape of the distribution.</p>

        <p>📊 Higher bars mean more students received that probability score.</p>

        <p>This visualization helps you understand how confident the model is about placement predictions for different students — whether most are highly likely, uncertain, or unlikely to be placed.</p>
        </div>
    """
    return fig_path, msg

# Scatter Plot: SSC Percentage vs MBA Percentage colored by Status
def plot_ssc_mba():
    testing_data, scaler, label_encoders, df, X, y = set_dataset()
    label_encoders, scaler, rf_model = load_models()
    X_train, X_test, y_train, y_test = testing_data

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ssc_p', y='hsc_p', hue=label_encoders['status'].inverse_transform(y), style=label_encoders['status'].inverse_transform(y))
    plt.title('SSC Percentage vs HSC Percentage by Placement Status')
    plt.xlabel('SSC Percentage')
    plt.ylabel('HSC Percentage')
    plt.legend(title='Status')
    fig_path = os.path.join(str(BASE_DIR), 'static', 'graph', "ssc_vs_hsc.png")
    plt.savefig(fig_path)
    plt.close()
    msg = """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h3>🖼️ About the Image: SSC vs HSC Percentage by Placement Status</h3>
        <p>The image displays a <strong>scatter plot</strong> titled <strong>"SSC Percentage vs HSC Percentage by Placement Status"</strong>.</p>

        <p>🔵 Each point represents a student, with:</p>
        <ul>
            <li>📊 The <strong>X-axis</strong> showing the student's <strong>SSC percentage</strong>.</li>
            <li>📏 The <strong>Y-axis</strong> showing the student's <strong>HSC percentage</strong>.</li>
        </ul>

        <p>🟢 The points are colored based on the student's <strong>placement status</strong> (whether they were placed or not), with different colors indicating the placement outcome.</p>

        <p>⚪ The <strong>styling</strong> of the points (shape) also differentiates placement statuses, allowing you to easily identify which students were placed and which weren't.</p>

        <p>📍 The scatter plot helps you visually analyze if there’s any trend or correlation between SSC and HSC percentages for students who were placed vs. those who weren’t.</p>
        </div>
    """
    return fig_path, msg

# Box Plot: Degree Percentage by Placement Status
def plot_degree_boxplot():
    testing_data, scaler, label_encoders, df, X, y = set_dataset()
    label_encoders, scaler, rf_model = load_models()
    X_train, X_test, y_train, y_test = testing_data

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=label_encoders['status'].inverse_transform(y), y=df['degree_p'])
    plt.title('Degree Percentage Distribution by Placement Status')
    plt.xlabel('Placement Status')
    plt.ylabel('Degree Percentage')
    fig_path = os.path.join(str(BASE_DIR), 'static', 'graph', "degree_boxplot.png")
    plt.savefig(fig_path)
    plt.close()
    msg = """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h3>🖼️ About the Image: Degree Percentage Distribution by Placement Status</h3>
        <p>The image displays a <strong>boxplot</strong> titled <strong>"Degree Percentage Distribution by Placement Status"</strong>.</p>

        <p>📊 <strong>Boxplot Representation</strong> – Shows the distribution of degree percentages across students, grouped by their <strong>placement status</strong> (whether they were placed or not).</p>

        <ul>
            <li>🏷️ <strong>X-axis:</strong> Represents the <strong>placement status</strong> (Placed vs Not Placed).</li>
            <li>📏 <strong>Y-axis:</strong> Shows the <strong>degree percentage</strong> of each student.</li>
        </ul>

        <p>📈 <strong>Box Characteristics:</strong></p>
        <ul>
            <li><strong>Box:</strong> Represents the <strong>interquartile range</strong> (IQR), showing the middle 50% of degree percentages for students in each placement group.</li>
            <li><strong>Whiskers:</strong> Extend to show the <strong>range of values</strong> within 1.5 times the IQR from the median.</li>
            <li><strong>Outliers:</strong> Any degree percentages outside the whiskers are marked as <strong>outliers</strong>.</li>
            <li><strong>Median line:</strong> Inside each box, a line represents the <strong>median degree percentage</strong>.</li>
        </ul>

        <p>This boxplot helps you understand how degree percentages vary between students who got placed and those who didn’t.</p>
    </div>
    """
    return fig_path, msg

# Correlation Heatmap
def plot_heatmap():
    testing_data, scaler, label_encoders, df, X, y = set_dataset()
    label_encoders, scaler, rf_model = load_models()
    X_train, X_test, y_train, y_test = testing_data

    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Features')
    plt.savefig('correlation_heatmap.png')
    fig_path = os.path.join(str(BASE_DIR), 'static', 'graph', "correlation_heatmap.png")
    plt.savefig(fig_path)
    plt.close()
    msg = """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h3>🖼️ About the Image: Correlation Heatmap of Features</h3>
        <p>The image displays a <strong>heatmap</strong> titled <strong>"Correlation Heatmap of Features"</strong>.</p>

        <p>📊 <strong>Heatmap Representation</strong> – Visualizes the <strong>correlation matrix</strong> of all the features used in the dataset, showing how strongly each pair of features is related.</p>

        <ul>
            <li>🏷️ <strong>Cells:</strong> Each cell represents the <strong>correlation</strong> between two features. The <strong>darker</strong> the color, the stronger the correlation between those features.</li>
            <li>📏 <strong>Color Scale:</strong> The color gradient goes from <strong>cool</strong> (blue) for negative correlations to <strong>warm</strong> (red) for positive correlations.</li>
            <li><strong>Positive Correlation:</strong> Features that move in the same direction.</li>
            <li><strong>Negative Correlation:</strong> Features that move in opposite directions.</li>
            <li><strong>Zero Correlation:</strong> Features that are unrelated to each other.</li>
        </ul>

        <p>🎯 <strong>Diagonal Cells:</strong> The diagonal of the heatmap will always be <strong>1</strong> since it represents the correlation of each feature with itself.</p>

        <p>This heatmap helps you quickly identify which features are highly correlated with each other, which can be useful for identifying multicollinearity or redundant features in your dataset.</p>
    </div>"""
    return fig_path, msg

