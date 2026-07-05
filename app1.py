import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config to wide mode for crisp visualizations
st.set_page_config(page_title="Laptop Price Predictor & Insights (USD)", layout="wide")

# Conversion rate factor (INR to USD)
INR_TO_USD_RATE = 83.5


# 1. DATA CACHING & LOADING PIPELINE

@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Handle rogue data strings like '?' found during EDA
    df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')
    df['Inches'] = df['Inches'].fillna(df['Inches'].median())
    
    # CONVERSION TO USD: Divide the raw price column by the exchange rate
    df['Price'] = (df['Price'] / INR_TO_USD_RATE).round(2)
    return df

# !!! UPDATE THIS PATH TO MATCH YOUR FILE LOCATION !!!
try:
    pc_price = load_and_clean_data("laptopData.csv")
except Exception as e:
    st.error("Could not find laptopData.csv at the specified path. Please configure your absolute path on line 29.")
    st.stop()


# 2. TRANSFORMATION & ENCODING STEP

@st.cache_resource
def preprocess_and_train(df):
    cols = ['Company', 'TypeName', 'Gpu', 'OpSys', 'Cpu']
    features_df = df[cols].copy()
    
    # Text extractions
    features_df['WeightWithoutUnit'] = df['Weight'].str.extract(r'(\d+\.?\d*)').astype(float)
    features_df['RamWithoutUnit'] = df['Ram'].str.extract(r'(\d+)').astype(int)
    features_df['SSD'] = df['Memory'].str.contains('SSD').astype(int)
    
    df_resolution = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
    features_df['Resolution_Width'] = df_resolution[0].astype(int)
    features_df['Resolution_Height'] = df_resolution[1].astype(int)
    features_df['Inches'] = df['Inches']

    # One-hot encoding categorical variables
    pc_price_encode = pd.get_dummies(features_df, dtype=int)
    
    # Impute missing extracts
    pc_price_encode['WeightWithoutUnit'] = pc_price_encode['WeightWithoutUnit'].fillna(pc_price_encode['WeightWithoutUnit'].median())
    pc_price_encode['RamWithoutUnit'] = pc_price_encode['RamWithoutUnit'].fillna(pc_price_encode['RamWithoutUnit'].median())
    
    # Target allocation
    X = pc_price_encode.copy()
    y = df['Price']  # Scaled to USD
    
    # Retain layout schema for downstream prediction mapping
    feature_columns = X.columns.tolist()
    
    # Splitting and Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
    s = StandardScaler().fit(X_train)
    X_train_scaled = s.transform(X_train)
    X_test_scaled = s.transform(X_test)
    
    # Dimensionality Reduction with PCA
    pca = PCA(n_components=10).fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Linear Regression Model
    lr = LinearRegression().fit(X_train_pca, y_train)
    pred_lr = lr.predict(X_test_pca)
    
    # Tuned Random Forest Regressor
    rf_grid = {'max_depth': [10, 15], 'min_samples_leaf': [2, 5], 'n_estimators': [100, 150]}
    rf_search = GridSearchCV(RandomForestRegressor(random_state=22), rf_grid, cv=3, scoring='r2', n_jobs=-1).fit(X_train_pca, y_train)
    rf_best = rf_search.best_estimator_
    pred_rf = rf_best.predict(X_test_pca)
    
    # Tuned Decision Tree Regressor
    dt_grid = {'max_depth': [5, 10], 'min_samples_leaf': [5, 10]}
    dt_search = GridSearchCV(DecisionTreeRegressor(random_state=22), dt_grid, cv=3, scoring='r2', n_jobs=-1).fit(X_train_pca, y_train)
    dt_best = dt_search.best_estimator_
    pred_dt = dt_best.predict(X_test_pca)
    
    return s, pca, lr, rf_best, dt_best, feature_columns, y_test, pred_lr, pred_rf, pred_dt, pc_price_encode

scaler, pca_transformer, model_lr, model_rf, model_dt, encoded_cols, y_test, p_lr, p_rf, p_dt, pc_price_encode = preprocess_and_train(pc_price)


# 3. STREAMLIT INTERFACE STRUCTURE

st.title("💻 Laptop Market Analytics & Machine Learning Pipeline (USD)")
st.markdown("An end-to-end framework parsing hardware configurations, converting Indian Rupee pricing tables natively to US Dollars ($), and running optimizations via PCA.")

# FIXED: Removed the unsupported 'key' keyword to maintain backward compatibility with your local library environment
tabs = st.tabs(["📋 Data Overview", "📊 Visual Explorations (EDA)", "⚙️ Preprocessing Summary", "🎯 Model Evaluation Metrics", "🔮 Live Value Predictor"])

# --- TAB 1: DATA OVERVIEW ---
with tabs[0]:
    st.header("Project Problem & Dataset Profile")
    st.write("This application focuses on regression-based predictions forecasting laptop market valuations using multi-dimensional specs. The pricing rows have been converted from native Indian Rupees to USD ($) using a 1 USD = 83.5 INR baseline.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Total Records Identified", pc_price.shape[0])
        st.metric("Base Features Captured", pc_price.shape[1])
    with col2:
        st.dataframe(pc_price.head(10), use_container_width=True)

# --- TAB 2: EXPLORATORY DATA ANALYSIS ---
with tabs[1]:
    st.header("Visual Explorations (EDA)")
    
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        heatmap_df = pd.DataFrame({
            'Cpu_Encoded': pc_price['Cpu'].map(pc_price.groupby('Cpu')['Price'].mean()),
            'Gpu_Encoded': pc_price['Gpu'].map(pc_price.groupby('Gpu')['Price'].mean()),
            'Price': pc_price['Price']
        })
        fig1 = px.imshow(heatmap_df.corr(), text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap: Component Impact")
        st.plotly_chart(fig1, use_container_width=True)
        st.info("I found that the CPU and GPU have a huge positive relationship together when deciding price tiers, which means high-end chips usually always get paired up with expensive hardware options.")
        
    with col_eda2:
        company_price = pc_price.groupby('Company', as_index=False)['Price'].mean().sort_values(by='Price', ascending=False)
        fig2 = px.bar(company_price, x='Company', y='Price', title='Average Price Benchmark Across Brands ($)', color='Price', color_continuous_scale='Turbo', labels={'Price': 'Average Price ($)'})
        st.plotly_chart(fig2, use_container_width=True)
        st.info("When looking at the brands, premium companies and unique software ecosystems clearly demand higher average baseline costs compared to brands that just make standard everyday consumer models.")

    st.markdown("---")
    col_eda3, col_eda4 = st.columns(2)
    with col_eda3:
        storage_analysis = pd.DataFrame({'Price': pc_price['Price'], 'Storage Type': pc_price['Memory'].str.contains('SSD').map({True: 'Contains SSD', False: 'HDD / Other Only'})})
        fig3 = px.box(storage_analysis, x='Storage Type', y='Price', color='Storage Type', title='Storage Architecture Price Impacts ($)', labels={'Price': 'Laptop Price ($)'})
        st.plotly_chart(fig3, use_container_width=True)
        st.info("Laptops with an SSD are significantly more expensive and have a much higher floor price than older mechanical HDD setups. Almost all high-end models completely dropped HDDs.")
        
    with col_eda4:
        fig4 = px.scatter(
            x=pc_price['Inches'], 
            y=pc_price_encode['Resolution_Width'], 
            color=pc_price['Price'], 
            title='Pixel Density Bounds vs Price Scaling ($)', 
            labels={'x': 'Inches', 'y': 'Resolution Width', 'color': 'Price ($)'}, 
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.info("Screen size alone doesn't actually change the laptop price much, but having a really high resolution screen inside a smaller body scales up the cost quickly because it targets luxury lines.")

    st.markdown("---")
    col_eda5, col_eda6 = st.columns(2)
    with col_eda5:
        opsys_price = pc_price.groupby('OpSys', as_index=False)['Price'].mean().sort_values(by='Price', ascending=False)
        fig5 = px.bar(opsys_price, x='OpSys', y='Price', title='Operating System License Markups ($)', color='Price', color_continuous_scale='Cividis')
        st.plotly_chart(fig5, use_container_width=True)
        st.info("Laptops that come pre-loaded with official enterprise operating systems cost noticeably more on average than blank setups with No OS, showing that licensing fees are included in the price.")
        
    with col_eda6:
        fig6 = px.scatter(
            x=pc_price_encode['WeightWithoutUnit'],
            y=pc_price['Price'],
            color=pc_price['TypeName'],
            title='The Portability vs Mass Pricing Distribution Balance',
            labels={'x': 'Weight (kg)', 'y': 'Price ($)'}
        )
        st.plotly_chart(fig6, use_container_width=True)
        st.info("The weight data shows two interesting groups: heavy laptops are expensive when they carry thick gaming graphics cards, while ultralight laptops are expensive because engineering compact thin designs costs extra.")

# --- TAB 3: PREPROCESSING SUMMARY ---
with tabs[2]:
    st.header("Feature Transformation Framework")
    st.markdown(f"""
    * **Currency Realignment:** Converted all row targets from INR to USD ($) using a fixed division filter.
    * **Regular Expression Extractions:** String indicators peeled into raw metrics (`Weight`, `Ram`, `Resolution_Width`, `Resolution_Height`).
    * **Categorical One-Hot Encoding:** Converting unstructured nominal labels via `pd.get_dummies()` resulted in a sparse **{pc_price_encode.shape[1] - 1} column matrix**.
    * **Standardization:** Normalized inputs with `StandardScaler` to force uniform zero-mean structures.
    * **Principal Component Analysis (PCA):** Compressed the high-dimensional features into **10 principal vectors**, mitigating multi-collinearity risks while holding prime structural variance.
    """)
    st.code("# Dimensionality Transformation Snippet\npca = PCA(n_components=10)\nX_train_pca = pca.fit_transform(X_train_scaled)")

# --- TAB 4: MODEL COMPARISON ---
with tabs[3]:
    st.header("Performance Matrices Comparison (USD)")
    
    def get_metrics(actual, pred):
        return round(mean_absolute_error(actual, pred), 2), round(mean_squared_error(actual, pred), 2), round(r2_score(actual, pred), 3)
    
    mae_lr, mse_lr, r2_lr = get_metrics(y_test, p_lr)
    mae_rf, mse_rf, r2_rf = get_metrics(y_test, p_rf)
    mae_dt, mse_dt, r2_dt = get_metrics(y_test, p_dt)
    
    metric_table = pd.DataFrame({
        "Algorithm Performance Metric": ["Mean Absolute Error (MAE in $)", "Mean Squared Error (MSE)", "Coefficient of Determination (R2 Score)"],
        "Linear Regression Baseline": [f"${mae_lr:,.2f}", mse_lr, r2_lr],
        "Optimized Random Forest": [f"${mae_rf:,.2f}", mse_rf, r2_rf],
        "Tuned Decision Tree Splitter": [f"${mae_dt:,.2f}", mse_dt, r2_dt]
    })
    st.table(metric_table)
    
    st.subheader("Residual Validation Curve (Random Forest)")
    eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': p_rf})
    fig_eval = px.scatter(eval_df, x='Actual', y='Predicted', trendline="ols", labels={'Actual': 'Observed Prices ($)', 'Predicted': 'Estimated Value Output ($)'})
    st.plotly_chart(fig_eval, use_container_width=True)
    st.info("**Insight 5:** The hyperparameter-tuned ensemble Random Forest model outperforms the others, dealing with non-linear feature interactions much better than a simple flat Linear Regression line.")

# --- TAB 5: LIVE USER PREDICTION ---
with tabs[4]:
    st.header("Real-Time Price Estimator Simulation")
    st.write("Adjust configuration profiles below and press the prediction button. Inputs are contained inside a standard form to secure persistent tab visibility.")
    
    # Wrapped inputs inside st.form to stop page refocusing bugs completely
    with st.form(key="prediction_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            sel_company = st.selectbox("Brand Name Profile", sorted(pc_price['Company'].unique()))
            sel_type = st.selectbox("Form Factor / Segment", sorted(pc_price['TypeName'].unique()))
            sel_opsys = st.selectbox("Operating System Deployment", sorted(pc_price['OpSys'].unique()))
        with col_in2:
            sel_cpu = st.selectbox("Processing Unit Variant", sorted(pc_price['Cpu'].unique()))
            sel_gpu = st.selectbox("Graphical Processor Unit Variant", sorted(pc_price['Gpu'].unique()))
            sel_ssd = st.radio("Primary Storage Configuration", ["Includes Solid-State Drive (SSD)", "Traditional Platters Only / HDD"])
        with col_in3:
            input_ram = st.slider("Memory Provisioning (RAM in GB)", int(pc_price_encode['RamWithoutUnit'].min()), int(pc_price_encode['RamWithoutUnit'].max()), 8)
            input_weight = st.slider("Chassis Mass (Weight in kg)", float(pc_price_encode['WeightWithoutUnit'].min()), float(pc_price_encode['WeightWithoutUnit'].max()), 2.0)
            input_inches = st.slider("Diagonal Screen Boundary (Inches)", float(pc_price['Inches'].min()), float(pc_price['Inches'].max()), 15.6)
            
        st.markdown("---")
        selected_model_name = st.selectbox(
            "Select Machine Learning Algorithm for Inference",
            ["Optimized Random Forest", "Tuned Decision Tree Splitter", "Linear Regression Baseline"]
        )
            
        # The submit button triggers calculation without resetting layout focus
        submit_button = st.form_submit_button(label="Generate Value Estimation Output")
        
    if submit_button:
        # Match width extractions from raw mappings
        match_res = pc_price[pc_price['Company'] == sel_company]['ScreenResolution'].iloc[0]
        res_extract = pd.Series([match_res]).str.extract(r'(\d+)x(\d+)')
        w_px, h_px = int(res_extract[0].iloc[0]), int(res_extract[1].iloc[0])

        # Build empty structure row matching encoding expectations exactly
        user_input_dict = {col: 0 for col in encoded_cols}
        
        # Inject numerical vectors
        user_input_dict['WeightWithoutUnit'] = input_weight
        user_input_dict['RamWithoutUnit'] = input_ram
        user_input_dict['SSD'] = 1 if "SSD" in sel_ssd else 0
        user_input_dict['Resolution_Width'] = w_px
        user_input_dict['Resolution_Height'] = h_px
        user_input_dict['Inches'] = input_inches
        
        # Activate matching dummy binary flags
        if f'Company_{sel_company}' in user_input_dict: user_input_dict[f'Company_{sel_company}'] = 1
        if f'TypeName_{sel_type}' in user_input_dict: user_input_dict[f'TypeName_{sel_type}'] = 1
        if f'OpSys_{sel_opsys}' in user_input_dict: user_input_dict[f'OpSys_{sel_opsys}'] = 1
        if f'Cpu_{sel_cpu}' in user_input_dict: user_input_dict[f'Cpu_{sel_cpu}'] = 1
        if f'Gpu_{sel_gpu}' in user_input_dict: user_input_dict[f'Gpu_{sel_gpu}'] = 1
        
        # Convert dictionary to DataFrame with correct column order
        input_df = pd.DataFrame([user_input_dict])[encoded_cols]
        
        # Transform inputs through training pipelines
        scaled_input = scaler.transform(input_df)
        pca_input = pca_transformer.transform(scaled_input)
        
        # Display predictions conditionally based on the chosen algorithm option
        st.markdown("### Estimation Results Analysis")
        
        if selected_model_name == "Optimized Random Forest":
            out_rf = model_rf.predict(pca_input)[0]
            st.metric("Optimized Random Forest Prediction", f"${max(0.0, round(out_rf, 2)):,.2f}")
            
        elif selected_model_name == "Tuned Decision Tree Splitter":
            out_dt = model_dt.predict(pca_input)[0]
            st.metric("Decision Tree Prediction", f"${max(0.0, round(out_dt, 2)):,.2f}")
            
        elif selected_model_name == "Linear Regression Baseline":
            out_lr = model_lr.predict(pca_input)[0]
            st.metric("Linear Regression Prediction", f"${max(0.0, round(out_lr, 2)):,.2f}")