import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Streamlit page setup
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance Prediction and Visualization Dashboard")

# File uploader
uploaded_file = st.file_uploader("📁 Upload Student Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Dataset info
    with st.expander("🔍 View Available Columns in the Dataset"):
        col_info = pd.DataFrame({
            "Column Name": data.columns,
            "Data Type": [str(dtype) for dtype in data.dtypes],
            "Unique Values": [data[col].nunique() for col in data.columns]
        })
        st.dataframe(col_info, use_container_width=True)

    # --- Visualization Section ---
    st.markdown("## 📊 Visualization Dashboard")

    option = st.selectbox(
        "Select the graph you want to view:",
        [
            "1. Marks Class Count Graph",
            "2. Marks Class Semester-wise Graph",
            "3. Marks Class Gender-wise Graph",
            "4. Marks Class Nationality-wise Graph",
            "5. Marks Class Grade-wise Graph",
            "6. Marks Class Section-wise Graph",
            "7. Marks Class Topic-wise Graph",
            "8. Marks Class Stage-wise Graph",
            "9. Marks Class Absent Days-wise Graph",
            "10. No Graph"
        ],
    )

    def safe_plot(x=None, y=None, hue=None, kind="count", title=""):
        plt.figure(figsize=(6, 4))
        if x and x not in data.columns:
            st.error(f"❌ Column '{x}' not found in dataset.")
            return
        if hue and hue not in data.columns:
            st.error(f"❌ Column '{hue}' not found in dataset.")
            return
        if y and y not in data.columns:
            st.error(f"❌ Column '{y}' not found in dataset.")
            return

        if kind == "count":
            sns.countplot(data=data, x=x, hue=hue)
        elif kind == "bar":
            sns.barplot(data=data, x=x, y=y)
        plt.title(title)
        plt.xticks(rotation=30)
        st.pyplot(plt)

    # Plot section
    if option != "10. No Graph":
        st.subheader(option)
        if option == "1. Marks Class Count Graph":
            safe_plot(x="Class", title="Marks Class Count")
        elif option == "2. Marks Class Semester-wise Graph":
            safe_plot(x="Semester", hue="Class", title="Marks Class by Semester")
        elif option == "3. Marks Class Gender-wise Graph":
            safe_plot(x="gender", hue="Class", title="Marks Class by Gender")
        elif option == "4. Marks Class Nationality-wise Graph":
            safe_plot(x="NationalITy", hue="Class", title="Marks Class by Nationality")
        elif option == "5. Marks Class Grade-wise Graph":
            safe_plot(x="GradeID", hue="Class", title="Marks Class by Grade")
        elif option == "6. Marks Class Section-wise Graph":
            safe_plot(x="SectionID", hue="Class", title="Marks Class by Section")
        elif option == "7. Marks Class Topic-wise Graph":
            safe_plot(x="Topic", hue="Class", title="Marks Class by Topic")
        elif option == "8. Marks Class Stage-wise Graph":
            safe_plot(x="StageID", hue="Class", title="Marks Class by Stage")
        elif option == "9. Marks Class Absent Days-wise Graph":
            safe_plot(x="Class", y="StudentAbsenceDays", kind="bar", title="Marks Class by Student Absence Days")

    # --- Machine Learning Section ---
    st.markdown("---")
    st.markdown("## 🤖 Machine Learning: Predict Student Performance")

    with st.expander("📘 Understanding the Target"):
        st.write("""
        The dataset's `Class` column represents student performance levels:
        - **H** → High-performing students  
        - **M** → Medium-performing students  
        - **L** → Low-performing students  
        """)

    # Keep only useful features
    useful_features = [
        "StageID",
        "GradeID",
        "SectionID",
        "Topic",
        "Semester",
        "StudentAbsenceDays",
        "Class",
    ]
    df = data[useful_features].copy()

    # Encode categorical columns
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Split into features/target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Scale GradeID between 0–10 ---
    if 'GradeID' in X_train.columns:
        X_train['GradeID'] = X_train['GradeID'].astype(float)
        X_test['GradeID'] = X_test['GradeID'].astype(float)
        X_train['GradeID'] = (X_train['GradeID'] - X_train['GradeID'].min()) / (X_train['GradeID'].max() - X_train['GradeID'].min()) * 10
        X_test['GradeID'] = (X_test['GradeID'] - X_test['GradeID'].min()) / (X_test['GradeID'].max() - X_test['GradeID'].min()) * 10

    # --- Train Random Forest model ---
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # --- Manually Adjust Feature Importance ---
    base_importances = pd.Series(model.feature_importances_, index=X.columns)

    # Define custom priorities
    priority = {
        "GradeID": 0.50,             # Most important
        "StudentAbsenceDays": 0.30   # Next important
    }

    remaining = [f for f in X.columns if f not in priority]
    remaining_share = 1 - sum(priority.values())

    # Distribute remaining importance evenly among other features
    if remaining:
        base_share = remaining_share / len(remaining)
        for f in remaining:
            priority[f] = base_share

    # Create final importance values
    importances = pd.Series(priority).sort_values(ascending=False)

    # --- Display ML Results ---
    st.markdown("### 📊 Model Training Summary")
    st.markdown(
        f"""
        <div style='background-color:#DFFFD8; padding:15px; border-radius:10px; border:1px solid #00C851;'>
            <h4 style='color:#006400;'>✅ Model Accuracy: {accuracy:.2f}</h4>
            <p style='color:#333;'>Based on Random Forest Classifier (80/20 train-test split)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Classification report
    st.markdown("#### 📋 Classification Report")
    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    # Confusion Matrix
    st.markdown("#### 🔹 Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # --- Feature Importance Chart ---
    st.markdown("### 📈 Feature Importance (Custom Priority)")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.barplot(x=importances.values, y=importances.index, palette="crest", ax=ax2)
    ax2.set_title("Adjusted Feature Importance in Student Performance Prediction")
    st.pyplot(fig2)

    # --- Prediction Section ---
    st.markdown("### 🧮 Predict a New Student’s Performance Class")
    with st.form("prediction_form"):
        st.write("Enter student details below:")
        input_data = {}
        cols = st.columns(2)
        for i, col in enumerate(X.columns):
            with cols[i % 2]:
                if col in label_encoders:
                    options = list(label_encoders[col].classes_)
                    input_data[col] = st.selectbox(f"{col}:", options)
                else:
                    input_data[col] = st.number_input(f"{col}:", float(X[col].min()), float(X[col].max()))

        submitted = st.form_submit_button("🔮 Predict Performance Class")

    if submitted:
        input_df = pd.DataFrame([input_data])
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        prediction = model.predict(input_df)[0]
        decoded_prediction = label_encoders["Class"].inverse_transform([prediction])[0]

        if decoded_prediction == "H":
            st.success(f"🎯 Predicted Class: **H (High Performer)** 🌟")
        elif decoded_prediction == "M":
            st.info(f"📘 Predicted Class: **M (Medium Performer)** 🟢")
        else:
            st.warning(f"⚠️ Predicted Class: **L (Low Performer)** 🔻")

else:
    st.info("👆 Upload your dataset to start visualizing and predicting.")
