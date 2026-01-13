# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Cervical Cancer 5-year OS Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Cervical Cancer 5-year OS Prediction System")
st.markdown("""
Cervical Cancer 5-year OS (Overall Survival) Prediction System Based on XGBoost, with SHAP interpretability analysis.
6 key clinical factors are used for prediction.
""")

# Set 6 key features
selected_features = [
    'LODDS',
    'FIGO_Stage',
    'TumorSize_cm',
    'Chemotherapy',
    'Marital_Status',
    'Surgery'
]

# Feature descriptions (修正医学逻辑错误)
feature_descriptions = {
    'LODDS': 'Lymph Node ODDS (log-transformed) - Higher values indicate worse prognosis',
    'FIGO_Stage': 'FIGO Stage - Higher stage indicates higher mortality risk (lower survival)',
    'TumorSize_cm': 'Tumor Size (cm) - Larger tumors decrease survival probability',  # 修正逻辑错误
    'Chemotherapy': 'Chemotherapy - Receiving chemotherapy reduces mortality risk (improves survival)',
    'Marital_Status': 'Marital Status - Married status associated with better prognosis',
    'Surgery': 'Surgery Type - Radical surgery reduces mortality risk compared to conservative surgery'
}


# ====================== 1. Load Model ======================
@st.cache_resource
def load_model():
    """Load model and preprocessing objects"""
    try:
        # 注意：如果实际用的是XGBoost，建议把文件名改为xgboost_model.pkl
        model = joblib.load('lightgbm_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Load feature info
        try:
            with open('feature_info.json', 'r', encoding='utf-8') as f:
                feature_info = json.load(f)
        except:
            feature_info = {
                'selected_features': selected_features,
                'feature_importance': [
                    {"feature": "LODDS", "importance": 0.4581},
                    {"feature": "FIGO_Stage", "importance": 0.3133},
                    {"feature": "TumorSize_cm", "importance": 0.2273},
                    {"feature": "Chemotherapy", "importance": 0.1827},
                    {"feature": "Marital_Status", "importance": 0.1265},
                    {"feature": "Surgery", "importance": 0.1130}
                ]
            }

        st.sidebar.success("✅ Model loaded successfully")
        return model, scaler, feature_info

    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        return None, None, {}


# Load model
model, scaler, feature_info = load_model()

# ====================== 2. Sidebar Navigation ======================
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Select Page",
    ["🔍 Single Sample Prediction", "📊 Feature Analysis", "ℹ️ User Guide"]
)

# ====================== 3. Single Sample Prediction Page ======================
if option == "🔍 Single Sample Prediction":
    st.header("Single Sample Prediction")
    st.markdown("Enter 6 key clinical factors to predict 5-year OS (Overall Survival) of cervical cancer.")

    # Create two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # 修复1：闭合subheader的字符串（核心语法错误）
        st.subheader("Risk-Increasing Factors (Higher values = higher mortality risk)")
        
        # 修复2：LODDS滑块参数规范化（负数范围+统一浮点数类型+匹配help文本）
        LODDS = st.slider(
            "LODDS (Lymph Node ODDS, log-transformed)",
            min_value=-2.2,  # 负数范围保留
            max_value=2.0,
            value=1.0,  # 改为浮点数，与min/max类型统一
            step=0.01,  # 更小的步长提升精度
            help="Typical range: -2.3 to 2.0, Mean: 0.5-1.0 (Higher = worse prognosis)"
        )

        FIGO_Stage = st.slider(
            "FIGO_Stage (Clinical Stage)",
            min_value=1.0, max_value=4.0, value=2.0, step=0.5,
            help="1=Early stage, 4=Advanced stage, Typical range: 1.0-4.0"
        )

        # 修复3：TumorSize_cm滑块范围修正（原0.0-1.0不合理）
        TumorSize_cm = st.slider(
            "TumorSize_cm (Tumor Size in cm)",
            min_value=1.0, max_value=2.0, value=1.0, step=1.0,
            help="2：≥4cm; 1:<4cm (Larger = worse prognosis)"
        )

    with col2:
        st.subheader("Risk-Reducing Factors (Higher values = lower mortality risk)")
        Chemotherapy = st.slider(
            "Chemotherapy (1=Received, 0=Not received)",
            min_value=0.0, max_value=1.0, value=1.0, step=1.0,
            help="1 = Patient received chemotherapy, 0 = No chemotherapy"
        )

        Marital_Status = st.slider(
            "Marital_Status (1=Married, 0=Unmarried)",
            min_value=0.0, max_value=1.0, value=1.0, step=1.0,
            help="1 = Married, 0 = Single/Divorced/Widowed"
        )

        Surgery = st.slider(
            "Surgery (1=Radical, 0=Conservative)",
            min_value=0.0, max_value=1.0, value=1.0, step=1.0,
            help="1 = Radical hysterectomy, 0 = Conservative surgery"
        )

    if st.button("🚀 Start Prediction", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded successfully, please check model files")
        else:
            with st.spinner("Analyzing..."):
                # Prepare input data
                input_data = {
                    'LODDS': LODDS,
                    'FIGO_Stage': FIGO_Stage,
                    'TumorSize_cm': TumorSize_cm,
                    'Chemotherapy': Chemotherapy,
                    'Marital_Status': Marital_Status,
                    'Surgery': Surgery
                }

                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)

                # Prediction
                try:
                    # Try different prediction methods
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(input_scaled)[0, 1]
                        print(f"Using predict_proba, probability: {probability}")
                    else:
                        # For native XGBoost/LightGBM Booster
                        try:
                            raw_pred = model.predict(input_scaled, output_margin=True)
                            if isinstance(raw_pred, np.ndarray) and len(raw_pred) > 0:
                                raw_score = float(raw_pred[0])
                            else:
                                raw_score = float(raw_pred)
                            probability = 1 / (1 + np.exp(-raw_score))
                            print(f"Using raw_score conversion, probability: {probability}")
                        except:
                            # Alternative method
                            pred = model.predict(input_scaled)
                            probability = float(pred[0]) if pred[0] <= 1 else 0.5
                            print(f"Using direct prediction, probability: {probability}")
                except Exception as pred_error:
                    st.error(f"Prediction failed: {pred_error}")
                    probability = 0.5

                # Ensure probability is between 0-1
                probability = max(0.0, min(1.0, probability))

                # 修复4：修正生存风险的逻辑表述（OS是生存率，概率越低风险越高）
                prediction = 1 if probability > 0.5 else 0
                prediction_label = "Low 5-Year Survival (High Risk)" if prediction == 1 else "High 5-Year Survival (Low Risk)"

                # Display prediction results
                st.subheader("📊 Prediction Results")

                # Create three-column layout for results
                col_result1, col_result2, col_result3 = st.columns(3)

                with col_result1:
                    if prediction == 1:
                        st.error(f"**Prediction Result: {prediction_label}**")
                    else:
                        st.success(f"**Prediction Result: {prediction_label}**")

                with col_result2:
                    # 修复5：明确是「死亡风险概率」或「生存概率」
                    st.metric("5-Year Mortality Risk Probability", f"{probability:.2%}")
                    st.metric("5-Year Survival Probability", f"{1-probability:.2%}")

                with col_result3:
                    # Risk level
                    if probability < 0.3:
                        risk_level = "Low Risk"
                        color = "green"
                    elif probability < 0.7:
                        risk_level = "Medium Risk"
                        color = "orange"
                    else:
                        risk_level = "High Risk"
                        color = "red"
                    st.markdown(f"**Risk Level**: :{color}[{risk_level}]")

                # Risk gauge chart（修正为死亡率仪表盘）
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "5-Year Mortality Risk (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ================= SHAP Interpretability Analysis =================
                st.subheader("🧠 Model Decision Explanation (SHAP Analysis)")

                try:
                    # Create SHAP explainer
                    background = np.zeros((10, len(selected_features)))
                    background_df = pd.DataFrame(background, columns=selected_features)
                    background_scaled = scaler.transform(background_df)

                    explainer = shap.TreeExplainer(model, background_scaled)
                    shap_values = explainer.shap_values(input_scaled)

                    # Handle SHAP output format
                    if isinstance(shap_values, list):
                        if len(shap_values) >= 2:
                            shap_vals = shap_values[1][0]
                        else:
                            shap_vals = shap_values[0][0]
                    else:
                        shap_vals = shap_values[0]

                    # Create SHAP values dataframe
                    shap_df = pd.DataFrame({
                        'Feature': selected_features,
                        'SHAP Value': shap_vals,
                        'Feature Value': input_df.iloc[0].values,
                        'Impact Direction': ['Increases Mortality Risk' if v > 0 else 'Decreases Mortality Risk' for v in shap_vals]
                    })

                    # Sort by absolute value
                    shap_df['Absolute Value'] = np.abs(shap_df['SHAP Value'])
                    shap_df = shap_df.sort_values('Absolute Value', ascending=False)

                    # Display SHAP table
                    st.dataframe(
                        shap_df[['Feature', 'Feature Value', 'SHAP Value', 'Impact Direction']].style.format({
                            'Feature Value': '{:.3f}',
                            'SHAP Value': '{:.4f}'
                        }),
                        use_container_width=True
                    )

                    # Visualize SHAP values
                    fig_shap = px.bar(shap_df,
                                      x='SHAP Value',
                                      y='Feature',
                                      orientation='h',
                                      color='Impact Direction',
                                      color_discrete_map={
                                          'Increases Mortality Risk': '#EF553B',
                                          'Decreases Mortality Risk': '#636EFA'
                                      },
                                      title='Feature Impact on Mortality Risk Prediction (SHAP Values)')

                    fig_shap.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
                    fig_shap.update_layout(height=400)
                    st.plotly_chart(fig_shap, use_container_width=True)

                    # Clinical interpretation
                    st.subheader("💡 Clinical Interpretation")

                    # Find top risk and protective factors
                    top_risk = shap_df[shap_df['SHAP Value'] > 0].head(2)
                    top_protective = shap_df[shap_df['SHAP Value'] < 0].head(2)

                    col_interpret1, col_interpret2 = st.columns(2)

                    with col_interpret1:
                        st.markdown("**Key Mortality Risk Factors:**")
                        if not top_risk.empty:
                            for _, row in top_risk.iterrows():
                                st.markdown(f"**{row['Feature']}** = {row['Feature Value']:.3f}")
                                st.markdown(f"- SHAP Value: **+{row['SHAP Value']:.4f}**")
                                st.markdown(f"- Explanation: {feature_descriptions.get(row['Feature'], '')}")
                        else:
                            st.info("No significant mortality risk factors")

                    with col_interpret2:
                        st.markdown("**Key Protective Factors (Improve Survival):**")
                        if not top_protective.empty:
                            for _, row in top_protective.iterrows():
                                st.markdown(f"**{row['Feature']}** = {row['Feature Value']:.3f}")
                                st.markdown(f"- SHAP Value: **{row['SHAP Value']:.4f}**")
                                st.markdown(f"- Explanation: {feature_descriptions.get(row['Feature'], '')}")
                        else:
                            st.info("No significant protective factors")

                    # Clinical recommendations（修正医学逻辑）
                    st.subheader("📋 Clinical Recommendations")
                    if probability > 0.7:
                        st.warning("""
                        **High Mortality Risk (>70%)**:
                        1. **Immediate Consultation**: Consult a gynecologic oncology specialist as soon as possible
                        2. **Enhanced Surveillance**: Increase frequency of follow-up examinations (every 3 months)
                        3. **Adjuvant Therapy**: Consider additional adjuvant therapy if not already administered
                        4. **Comprehensive Evaluation**: Complete PET-CT to rule out distant metastasis
                        """)
                    elif probability > 0.3:
                        st.warning("""
                        **Medium Mortality Risk (30%-70%)**:
                        1. **Specialist Follow-up**: Regular follow-up with gynecologic oncology team
                        2. **Standard Surveillance**: Follow-up every 6 months with imaging studies
                        3. **Lifestyle Modification**: Maintain healthy lifestyle and regular exercise
                        4. **Symptom Monitoring**: Prompt medical attention for any new symptoms
                        """)
                    else:
                        st.success("""
                        **Low Mortality Risk (<30%)**:
                        1. **Routine Surveillance**: Follow standard cervical cancer follow-up guidelines
                        2. **Annual Screening**: Annual gynecologic examination and imaging
                        3. **Health Maintenance**: Continue regular health check-ups and healthy lifestyle
                        4. **Patient Education**: Educate patient on warning signs of recurrence
                        """)

                except Exception as e:
                    st.error(f"SHAP analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("Model prediction function is normal, but interpretability analysis is temporarily unavailable")

# ====================== 4. Feature Analysis Page ======================
elif option == "📊 Feature Analysis":
    st.header("Feature Analysis")

    tab1, tab2 = st.tabs(["📈 Feature Importance", "ℹ️ Feature Explanations"])

    with tab1:
        st.subheader("Feature Importance Analysis")

        # Method 1: Get importance from feature_info.json
        if 'feature_importance' in feature_info:
            importance_data = feature_info['feature_importance']
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('importance', ascending=True)

            fig = px.bar(importance_df,
                         x='importance',
                         y='feature',
                         orientation='h',
                         title="Feature Importance Ranking (Based on XGBoost Gain)",
                         color='importance',
                         color_continuous_scale='Viridis')

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Feature Importance Explanation**:
            - Importance is calculated based on XGBoost gain (total contribution of feature to model)
            - Higher values indicate greater impact on prediction outcomes
            - Positive correlation means feature increases mortality risk, negative means decreases risk
            """)

        # Method 2: Try to get importance from XGBoost/LightGBM Booster
        elif model is not None:
            try:
                # Compatible with XGBoost/LightGBM
                if hasattr(model, 'get_score'):
                    importance_dict = model.get_score(importance_type='gain')
                    importances = [importance_dict.get(f, 0) for f in selected_features]
                elif hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    try:
                        importances = model.get_booster().get_score(importance_type='gain')
                        importances = [importances.get(f, 0) for f in selected_features]
                    except:
                        importances = np.ones(len(selected_features))

                importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': importances
                }).sort_values('importance', ascending=True)

                fig = px.bar(importance_df,
                             x='importance',
                             y='feature',
                             orientation='h',
                             title="Model Feature Importance (Gain)",
                             color='importance',
                             color_continuous_scale='Viridis')

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Unable to retrieve feature importance: {e}")

                # Show default importance
                st.info("Displaying default feature importance based on clinical relevance")
                default_importance = {
                    'LODDS': 0.4581,
                    'FIGO_Stage': 0.3133,
                    'TumorSize_cm': 0.2273,
                    'Chemotherapy': 0.1827,
                    'Marital_Status': 0.1265,
                    'Surgery': 0.1130
                }

                importance_df = pd.DataFrame({
                    'feature': list(default_importance.keys()),
                    'importance': list(default_importance.values())
                }).sort_values('importance', ascending=True)

                fig = px.bar(importance_df,
                             x='importance',
                             y='feature',
                             orientation='h',
                             title="Feature Importance (Based on Clinical Relevance)",
                             color='importance',
                             color_continuous_scale='Viridis')

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Feature importance information unavailable")

    with tab2:
        st.subheader("Detailed Feature Explanations")

        # Create feature information table
        feature_info_data = []
        for feature in selected_features:
            feature_info_data.append({
                'Feature': feature,
                'Description': feature_descriptions.get(feature, 'No description available'),
                'Relationship to Mortality': 'Positive' if feature in ['LODDS', 'FIGO_Stage', 'TumorSize_cm'] else 'Negative',
                'Typical Range': {
                    'LODDS': '-2.3-2.0',  # 匹配滑块的负数范围
                    'FIGO_Stage': '1.0-4.0',
                    'TumorSize_cm': '0.5-8.0',
                    'Chemotherapy': '0.0-1.0',
                    'Marital_Status': '0.0-1.0',
                    'Surgery': '0.0-1.0'
                }.get(feature, 'Unknown'),
                'XGBoost Gain': {
                    'LODDS': '+0.458',
                    'FIGO_Stage': '+0.313',
                    'TumorSize_cm': '+0.227',
                    'Chemotherapy': '-0.183',
                    'Marital_Status': '-0.127',
                    'Surgery': '-0.113'
                }.get(feature, 'Unknown')
            })

        feature_info_df = pd.DataFrame(feature_info_data)
        st.dataframe(feature_info_df, use_container_width=True)

        # Detailed explanations
        st.markdown("""
        ### 🎯 Clinical Significance of Features

        1. **LODDS (Lymph Node ODDS, log-transformed)**
           - **XGBoost Gain**: +0.458 (Most important survival predictor)
           - **Clinical Significance**: Measures lymph node involvement extent (log scale)
           - **Mortality Feature**: Higher LODDS indicates more extensive lymph node metastasis
           - **Reference Values**: 
             - Low risk: < 0.5
             - High risk: > 1.0

        2. **FIGO_Stage (International Federation of Gynecology and Obstetrics Stage)**
           - **XGBoost Gain**: +0.313
           - **Clinical Significance**: Standard staging system for cervical cancer
           - **Mortality Feature**: Higher stage indicates more advanced disease
           - **Staging Explanation**: 
             - Stage 1: Confined to cervix
             - Stage 4: Distant metastasis

        3. **TumorSize_cm (Tumor Size in Centimeters)**
           - **XGBoost Gain**: +0.227
           - **Clinical Significance**: Maximum dimension of primary tumor
           - **Mortality Feature**: Larger tumors have higher metastatic potential
           - **Clinical Threshold**: Tumors > 4cm have significantly higher mortality risk

        4. **Chemotherapy (Receipt of Chemotherapy)**
           - **XGBoost Gain**: -0.183 (Strong protective factor)
           - **Clinical Significance**: Systemic treatment to kill cancer cells
           - **Protective Effect**: Adjuvant chemotherapy reduces micrometastasis risk
           - **Implementation**: Typically administered post-surgery for high-risk patients

        5. **Marital_Status (Marital Status)**
           - **XGBoost Gain**: -0.127
           - **Clinical Significance**: Social support indicator
           - **Protective Effect**: Married patients often have better treatment adherence and support
           - **Note**: Sociodemographic factor, not direct biological predictor

        6. **Surgery (Surgery Type)**
           - **XGBoost Gain**: -0.113
           - **Clinical Significance**: Extent of surgical resection
           - **Protective Effect**: Radical hysterectomy removes more tissue than conservative surgery
           - **Options**: 
             - Radical: Removal of uterus, cervix, parametrium, and upper vagina
             - Conservative: Wide local excision preserving fertility
        """)

# ====================== 5. User Guide Page ======================
elif option == "ℹ️ User Guide":
    st.header("User Guide")

    st.markdown("""
    ## 📖 Cervical Cancer 5-Year OS Prediction System User Manual

    ### 1. System Overview
    This system is based on the XGBoost machine learning model, trained on cervical cancer clinical datasets.
    It predicts 5-year overall survival (OS) of cervical cancer using 6 key clinical factors and provides SHAP interpretability analysis.

    ### 2. Main Functions

    #### 🔍 Single Sample Prediction
    - **Function**: Predict 5-year survival probability for individual patients based on input features
    - **Steps**:
      1. Select "Single Sample Prediction" from the sidebar
      2. Adjust slider values for the 6 clinical factors
      3. Click the "Start Prediction" button
    - **Output**:
      - Prediction Result (Low/High 5-Year Survival)
      - 5-Year Mortality/Survival Probability
      - Risk Level Classification
      - SHAP Interpretability Analysis
      - Clinical Recommendations

    #### 📊 Feature Analysis
    - **Function**: Analyze feature importance in the prediction model
    - **Includes**:
      - XGBoost-based feature importance ranking
      - Detailed clinical explanations for each feature

    ### 3. Explanation of 6 Key Features

    The model uses 6 key clinical factors selected through feature importance analysis:

    | Feature | Relationship to Mortality | XGBoost Gain | Clinical Significance |
    |---------|--------------|--------------|-----------------------|
    | `LODDS` | Positive | +0.458 | Lymph Node ODDS (log), most important survival predictor |
    | `FIGO_Stage` | Positive | +0.313 | Clinical stage, higher stage = higher mortality risk |
    | `TumorSize_cm` | Positive | +0.227 | Tumor size, larger tumors increase mortality risk |
    | `Chemotherapy` | Negative | -0.183 | Receiving chemotherapy reduces mortality risk |
    | `Marital_Status` | Negative | -0.127 | Married status associated with better prognosis |
    | `Surgery` | Negative | -0.113 | Radical surgery reduces mortality risk |

    ### 4. Result Interpretation Guide

    #### Risk Level Classification:
    - **Low Risk (<30%)**: Low mortality probability, recommend routine follow-up
    - **Medium Risk (30%-70%)**: Need further evaluation, recommend specialist consultation
    - **High Risk (>70%)**: High mortality probability, recommend immediate medical attention

    #### SHAP Value Interpretation:
    - **Positive SHAP Value**: Increases mortality risk
    - **Negative SHAP Value**: Decreases mortality risk
    - **Absolute Value Magnitude**: Indicates strength of impact

    ### 5. Technical Information

    - **Model Algorithm**: XGBoost (Extreme Gradient Boosting)
    - **Training Data**: Cervical cancer clinical dataset
    - **Feature Selection**: Based on XGBoost feature gain (>0.1)
    - **Number of Features**: 6 key clinical factors
    - **Interpretability**: SHAP (SHapley Additive exPlanations)

    ### 6. Important Disclaimer

    ⚠️ **Disclaimer**:
    1. This system is a decision support tool and cannot replace professional medical diagnosis
    2. All prediction results are for reference only
    3. Clinical decisions should be based on complete clinical information
    4. Consult medical professionals for any questions or concerns

    ### 7. Troubleshooting

    **Common Issues**:
    1. **Model Loading Failure**: Check if `xgboost_model.pkl` (or lightgbm_model.pkl) and `scaler.pkl` files exist
    2. **SHAP Analysis Failure**: May be due to insufficient memory, try restarting the application
    3. **Abnormal Prediction Results**: Check if input feature values are within reasonable ranges

    **Technical Support**: Contact system administrator for any technical issues
    """)

# ====================== 6. Footer ======================
st.sidebar.markdown("---")
st.sidebar.info("""
**Cervical Cancer 5-Year OS Prediction System**  

🏥 Machine Learning-based Clinical Decision Support Tool  
📊 With SHAP Interpretability Analysis  
⚠️ For Medical Professional Use Only (Reference Purpose)
""")

# Add refresh button
if st.sidebar.button("🔄 Refresh Application"):
    st.rerun()



