import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

sns.set_style('whitegrid')


st.set_page_config(page_title="Data Analysis", layout="wide")

st.title("📈 Data Analysis and Visualization Web Application")
st.subheader("Designed by Sanskar Sharma")


# Function to download plots
def download_button(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("Download Graph 📥", data=buf.getvalue(), file_name=filename, mime="image/png")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='unicode_escape')

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Data Cleaning
    df.drop(['Status', 'unnamed1'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df['Amount'] = df['Amount'].astype(int)
    df.rename(columns={'Marital_Status': 'Shaadi'}, inplace=True)

    st.subheader("Data Summary")
    st.write(df.describe())

    # Sidebar filters
    st.sidebar.header("Filter Options")
    gender = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    age_group = st.sidebar.multiselect("Select Age Group", options=df['Age Group'].unique(), default=df['Age Group'].unique())

    filtered_df = df[(df['Gender'].isin(gender)) & (df['Age Group'].isin(age_group))]

    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

# Machine Learning Part (Simple Prediction)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.subheader("🔮 Machine Learning: Predict Sales Amount")

# Selecting features and target
features = filtered_df[['Gender', 'Age Group', 'Occupation']]
target = filtered_df['Amount']

# Encode categorical variables
le = LabelEncoder()
for col in features.columns:
    features[col] = le.fit_transform(features[col])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Score
st.write(f"Model Training Score: {model.score(X_train, y_train)*100:.2f}%")

# Making a simple prediction
sample_input = {
    'Gender': st.selectbox("Select Gender", options=filtered_df['Gender'].unique()),
    'Age Group': st.selectbox("Select Age Group", options=filtered_df['Age Group'].unique()),
    'Occupation': st.selectbox("Select Occupation", options=filtered_df['Occupation'].unique())
}

# Encode sample input
sample_df = pd.DataFrame([sample_input])
for col in sample_df.columns:
    sample_df[col] = le.fit_transform(sample_df[col])

# Predict
predicted_amount = model.predict(sample_df)[0]
st.success(f"Predicted Sales Amount: ₹{int(predicted_amount):,}")

# 👇 After this, your existing "Visualizations" section will continue


    # Visualizations
    st.subheader("Visualizations")

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Gender Distribution")
        with st.spinner('Loading chart...'):
            fig, ax = plt.subplots(figsize=(8,5))
            ax = sns.countplot(x='Gender', data=filtered_df, palette='Set2')  # Set color palette
            for bars in ax.containers:
                ax.bar_label(bars)
            st.pyplot(fig)
            download_button(fig, "gender_distribution.png")

    with col2:
        st.markdown("### Gender vs Total Amount")
        with st.spinner('Loading chart...'):
            sales_gen = filtered_df.groupby(['Gender'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x='Gender', y='Amount', data=sales_gen, palette='Set2')  # Set color palette
            st.pyplot(fig)
            download_button(fig, "gender_vs_amount.png")

    # Row 2
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Age Group vs Gender")
        with st.spinner('Loading chart...'):
            fig, ax = plt.subplots(figsize=(8,5))
            sns.countplot(data=filtered_df, x='Age Group', hue='Gender', ax=ax, palette='Set2')  # Set color palette
            for bars in ax.containers:
                ax.bar_label(bars)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            download_button(fig, "age_group_vs_gender.png")

    with col2:
        st.markdown("### Age Group vs Total Amount")
        with st.spinner('Loading chart...'):
            sales_age = filtered_df.groupby(['Age Group'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x='Age Group', y='Amount', data=sales_age, palette='Set2')  # Set color palette
            plt.xticks(rotation=45)
            st.pyplot(fig)
            download_button(fig, "age_group_vs_amount.png")

    # Row 3
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top 10 States by Orders")
        with st.spinner('Loading chart...'):
            sales_state = filtered_df.groupby(['State'], as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(data=sales_state, x='State', y='Orders', ax=ax, palette='Set2')  # Set color palette
            plt.xticks(rotation=90)
            st.pyplot(fig)
            download_button(fig, "top_states_by_orders.png")

    with col2:
        st.markdown("### Top 10 States by Amount")
        with st.spinner('Loading chart...'):
            sales_state = filtered_df.groupby(['State'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(data=sales_state, x='State', y='Amount', ax=ax, palette='Set2')  # Set color palette
            plt.xticks(rotation=90)
            st.pyplot(fig)
            download_button(fig, "top_states_by_amount.png")

    # Row 4
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Marital Status Distribution")
        with st.spinner('Loading chart...'):
            fig, ax = plt.subplots(figsize=(8,5))
            ax = sns.countplot(data=filtered_df, x='Shaadi', palette='Set2')  # Set color palette
            for bars in ax.containers:
                ax.bar_label(bars)
            st.pyplot(fig)
            download_button(fig, "marital_status_distribution.png")

    with col2:
        st.markdown("### Marital Status vs Amount by Gender")
        with st.spinner('Loading chart...'):
            sales_state = filtered_df.groupby(['Shaadi', 'Gender'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(data=sales_state, x='Shaadi', y='Amount', hue='Gender', ax=ax, palette='Set2')  # Set color palette
            st.pyplot(fig)
            download_button(fig, "marital_status_vs_amount_by_gender.png")

    # Row 5
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Occupation Distribution")
        with st.spinner('Loading chart...'):
            fig, ax = plt.subplots(figsize=(10,6))
            ax = sns.countplot(data=filtered_df, x='Occupation', palette='Set2')  # Set color palette
            for bars in ax.containers:
                ax.bar_label(bars)
            plt.xticks(rotation=90)
            st.pyplot(fig)
            download_button(fig, "occupation_distribution.png")

    with col2:
        st.markdown("### Occupation vs Amount")
        with st.spinner('Loading chart...'):
            sales_state = filtered_df.groupby(['Occupation'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(data=sales_state, x='Occupation', y='Amount', ax=ax, palette='Set2')  # Set color palette
            plt.xticks(rotation=90)
            st.pyplot(fig)
            download_button(fig, "occupation_vs_amount.png")

    # Row 6
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Product Category Distribution")
        with st.spinner('Loading chart...'):
            fig, ax = plt.subplots(figsize=(10,6))
            ax = sns.countplot(data=filtered_df, x='Product_Category', palette='Set2')  # Set color palette
            for bars in ax.containers:
                ax.bar_label(bars)
            plt.xticks(rotation=90)
            st.pyplot(fig)
            download_button(fig, "product_category_distribution.png")

    with col2:
        st.markdown("### Top 10 Product Categories by Amount")
        with st.spinner('Loading chart...'):
            sales_state = filtered_df.groupby(['Product_Category'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(data=sales_state, x='Product_Category', y='Amount', ax=ax, palette='Set2')  # Set color palette
            plt.xticks(rotation=90)
            st.pyplot(fig)
            download_button(fig, "top_product_categories_by_amount.png")

    # Row 7
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top 10 Products by Orders")
        with st.spinner('Loading chart...'):
            sales_state = filtered_df.groupby(['Product_ID'], as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(data=sales_state, x='Product_ID', y='Orders', ax=ax, palette='Set2')  # Set color palette
            plt.xticks(rotation=90)
            st.pyplot(fig)
            download_button(fig, "top_products_by_orders.png")

    with col2:
        st.markdown("### Top 10 Most Sold Products (Alternative View)")
        with st.spinner('Loading chart...'):
            fig, ax = plt.subplots(figsize=(10,6))
            filtered_df.groupby('Product_ID')['Orders'].sum().nlargest(10).sort_values(ascending=False).plot(kind='bar', ax=ax, color=sns.color_palette("Set2", 10))
            plt.xticks(rotation=90)
            st.pyplot(fig)
            download_button(fig, "top_products_alternative.png")

else:
    st.info("Please upload a CSV file to get started.")
