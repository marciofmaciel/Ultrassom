# ultrasonic_app_ml.py
"""
Aplicação Web com Machine Learning
Análise Ultrassônica + Previsão de Propriedades
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ml_predictor import ElasticPropertiesPredictor
import os

# Configuração da página
st.set_page_config(
    page_title="Ultrasonic Analysis + ML",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'training_data' not in st.session_state:
    st.session_state.training_data = None

# Header
st.markdown('<h1 class="main-header">🔬 Ultrasonic Analysis + ML Prediction</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Seção ML
    st.subheader("🤖 Machine Learning")
    
    st.markdown("---")
    st.subheader("📥 Dados Reais para Treinamento")
    real_data_file = st.file_uploader("Upload CSV com dados reais (colunas: rho, thickness, temperature, V_L_0, V_L_90, V_L_Z, V_T_0, V_T_90, V_T_Z, E1, E2, E3, G12, G13, G23)", type=["csv"])
    real_data = None
    if real_data_file is not None:
        try:
            real_data = pd.read_csv(real_data_file)
            st.success(f"Arquivo carregado: {real_data_file.name} ({real_data.shape[0]} linhas)")
            st.dataframe(real_data.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")

    if st.button("🎓 Train ML Models", use_container_width=True):
        with st.spinner("Training models... This may take a minute..."):
            predictor = ElasticPropertiesPredictor()
            synthetic_data = predictor.generate_synthetic_data(n_samples=1000)
            if real_data is not None:
                # Concatenar dados reais e sintéticos
                data = pd.concat([synthetic_data, real_data], ignore_index=True)
                st.info(f"Treinando com {synthetic_data.shape[0]} sintéticos + {real_data.shape[0]} reais")
            else:
                data = synthetic_data
            # Remover linhas com NaN nas variáveis alvo
            target_cols = ['E1', 'E2', 'E3', 'G12', 'G13', 'G23']
            n_before = data.shape[0]
            data = data.dropna(subset=target_cols)
            n_after = data.shape[0]
            if n_after < n_before:
                st.warning(f"{n_before - n_after} linhas removidas por NaN nas variáveis alvo.")
            # Treinar os 3 cenários padrão
            predictor.train_models(data, feature_set='full')
            predictor.train_models(data, feature_set='partial_L')
            predictor.train_models(data, feature_set='partial_minimal')
            # Treinar o novo cenário minimal_z sem sobrescrever prepare_features
            def prepare_minimal_z(data, feature_set=None):
                features = ['rho', 'thickness', 'temperature', 'V_L_Z', 'V_T_Z']
                predictor.feature_names = features
                return data[features]
            X_minimal_z = prepare_minimal_z(data)
            y_minimal_z = data[predictor.target_names]
            from sklearn.preprocessing import StandardScaler
            scaler_minimal_z = StandardScaler()
            X_minimal_z_scaled = scaler_minimal_z.fit_transform(X_minimal_z)
            predictor.scalers['minimal_z'] = scaler_minimal_z
            # Treinar modelos para minimal_z
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.neural_network import MLPRegressor
            from sklearn.multioutput import MultiOutputRegressor
            models_to_test = {
                'Random Forest': RandomForestRegressor(
                    n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
                'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
                'Neural Network': MultiOutputRegressor(MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=500, random_state=42))
            }
            best_score = -np.inf
            best_model_name = None
            results_minimal_z = {}
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import mean_squared_error, r2_score
            X_train, X_test, y_train, y_test = train_test_split(
                X_minimal_z, y_minimal_z, test_size=0.2, random_state=42)
            # Remover linhas com NaN nas variáveis alvo para minimal_z
            mask_valid = ~y_train.isnull().any(axis=1)
            X_train_scaled = scaler_minimal_z.transform(X_train[mask_valid])
            y_train_valid = y_train[mask_valid]
            mask_valid_test = ~y_test.isnull().any(axis=1)
            X_test_scaled = scaler_minimal_z.transform(X_test[mask_valid_test])
            y_test_valid = y_test[mask_valid_test]
            for name, model in models_to_test.items():
                model.fit(X_train_scaled, y_train_valid)
                y_pred_test = model.predict(X_test_scaled)
                r2_test = r2_score(y_test_valid, y_pred_test)
                rmse_test = np.sqrt(mean_squared_error(y_test_valid, y_pred_test))
                cv_scores = cross_val_score(model, X_train_scaled, y_train_valid, cv=5, scoring='r2', n_jobs=-1)
                results_minimal_z[name] = {
                    'model': model,
                    'r2_test': r2_test,
                    'rmse_test': rmse_test,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                if r2_test > best_score:
                    best_score = r2_test
                    best_model_name = name
            predictor.models['minimal_z'] = results_minimal_z[best_model_name]['model']
            # Salvar
            predictor.save_models('models.pkl')
            st.session_state.predictor = predictor
            st.session_state.models_trained = True
            st.session_state.training_data = data
        st.success("✓ Models trained successfully!")
    
    if os.path.exists('models.pkl') and not st.session_state.models_trained:
        if st.button("📂 Load Existing Models", use_container_width=True):
            predictor = ElasticPropertiesPredictor()
            predictor.load_models('models.pkl')
            st.session_state.predictor = predictor
            st.session_state.models_trained = True
            st.success("✓ Models loaded!")
    
    st.markdown("---")
    
    # Status
    if st.session_state.models_trained:
        st.success("🟢 ML Models Ready")
    else:
        st.warning("🟡 ML Models Not Loaded")

# Tabs principais
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Input", 
    "🤖 ML Prediction", 
    "📈 Model Performance",
    "🔍 Feature Analysis"
])

# ============================================================================
# TAB 1: DATA INPUT
# ============================================================================

with tab1:
    st.header("📊 Input Measurements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rho = st.number_input("Density (kg/m³)", value=1573, min_value=1000, max_value=2000)
    with col2:
        thickness = st.number_input("Thickness (mm)", value=4.87, min_value=1.0, max_value=20.0)
    with col3:
        temperature = st.number_input("Temperature (°C)", value=21, min_value=0, max_value=50)
    
    st.markdown("### Velocity Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Longitudinal Waves (m/s)**")
        V_L_0 = st.number_input("V_L (0°)", value=3498.0, key='vl0')
        V_L_90 = st.number_input("V_L (90°)", value=2001.0, key='vl90')
        V_L_Z = st.number_input("V_L (Z)", value=3001.0, key='vlz')
    
    with col2:
        st.markdown("**Transverse Waves (m/s)**")
        V_T_0 = st.number_input("V_T (0°)", value=1847.0, key='vt0')
        V_T_90 = st.number_input("V_T (90°)", value=1423.0, key='vt90')
        V_T_Z = st.number_input("V_T (Z)", value=1589.0, key='vtz')
    
    # Armazenar no session state
    st.session_state.input_data = {
        'rho': rho,
        'thickness': thickness,
        'temperature': temperature,
        'V_L_0': V_L_0,
        'V_L_90': V_L_90,
        'V_L_Z': V_L_Z,
        'V_T_0': V_T_0,
        'V_T_90': V_T_90,
        'V_T_Z': V_T_Z
    }

# Função utilitária para garantir todas as features necessárias

def fill_features(input_dict, feature_names):
    filled = {k: input_dict.get(k, np.nan) for k in feature_names}
    missing = [k for k in feature_names if k not in input_dict]
    if missing:
        st.warning(f"As seguintes features estavam faltando e foram preenchidas com np.nan: {missing}")
    return filled

# ============================================================================
# TAB 2: ML PREDICTION
# ============================================================================

with tab2:
    st.header("🤖 Machine Learning Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train or load ML models first (see sidebar)")
    else:
        predictor = st.session_state.predictor
        # Garantir que os dados de entrada do usuário sejam usados SEMPRE
        input_data = dict(st.session_state.input_data)
        
        # Seletor de cenário
        scenario = st.selectbox(
            "Select Prediction Scenario",
            [
                "Full Measurements (All velocities)",
                "Partial - Longitudinal Only",
                "Minimal - Only V_L(0°) and V_L(90°)",
                "Minimal - Only V_L(Z) and V_T(Z)"
            ]
        )
        
        feature_set_map = {
            "Full Measurements (All velocities)": 'full',
            "Partial - Longitudinal Only": 'partial_L',
            "Minimal - Only V_L(0°) and V_L(90°)": 'partial_minimal',
            "Minimal - Only V_L(Z) and V_T(Z)": 'minimal_z'
        }
        
        feature_set = feature_set_map[scenario]
        
        # Definir as features corretas para cada cenário
        feature_sets_features = {
            'full': ['rho', 'thickness', 'temperature', 'V_L_0', 'V_L_90', 'V_L_Z', 'V_T_0', 'V_T_90', 'V_T_Z'],
            'partial_L': ['rho', 'thickness', 'temperature', 'V_L_0', 'V_L_90', 'V_L_Z'],
            'partial_minimal': ['rho', 'thickness', 'temperature', 'V_L_0', 'V_L_90'],
            'minimal_z': ['rho', 'thickness', 'temperature', 'V_L_Z', 'V_T_Z']
        }
        
        # Seletor de modelo
        model_options = ['Random Forest', 'Gradient Boosting', 'Neural Network']
        selected_model = st.selectbox(
            "Select ML Model",
            model_options,
            index=0
        )
        
        if st.button("🔮 Predict Properties", type="primary", use_container_width=True):
            with st.spinner("Making predictions..."):
                feature_names = feature_sets_features[feature_set]
                # Sempre usar os valores informados pelo usuário, sem sobrescrever
                input_data_filled = {k: input_data.get(k, np.nan) for k in feature_names}
                # Usar o modelo já treinado e salvo
                predictions = predictor.predict_with_confidence(
                    input_data_filled, 
                    feature_set=feature_set
                )
                st.session_state.predictions = predictions
                st.session_state.input_data_filled = input_data_filled
                # Calcular e exibir Poisson (rho)
                try:
                    E2 = predictions['E2']['mean']
                    G12 = predictions['G12']['mean']
                    if G12 != 0:
                        nu12 = E2 / (2 * G12) - 1
                        st.info(f"Coeficiente de Poisson estimado (ν₁₂): {nu12:.3f}")
                    else:
                        st.warning("G12 = 0, não é possível calcular o coeficiente de Poisson.")
                except Exception as e:
                    st.warning(f"Não foi possível calcular o coeficiente de Poisson: {e}")
        
        # Mostrar resultados
        if 'predictions' in st.session_state:
            st.markdown("### 📝 Dados de Entrada Utilizados na Predição")
            # Exibir os dados de entrada realmente usados na predição
            input_data_filled = st.session_state.get('input_data_filled', {})
            if input_data_filled:
                st.dataframe(pd.DataFrame([input_data_filled]), use_container_width=True)
            else:
                st.info("Realize uma predição para visualizar os dados de entrada utilizados.")

            st.markdown("### 📊 Predicted Elastic Properties")
            predictions = st.session_state.predictions
            # Criar tabela
            results_data = []
            for prop, stats in predictions.items():
                results_data.append({
                    'Property': prop,
                    'Predicted Value (GPa)': f"{stats['mean']:.2f}",
                    '95% CI Lower': f"{stats['ci_lower']:.2f}",
                    '95% CI Upper': f"{stats['ci_upper']:.2f}",
                    'Uncertainty (±)': f"{stats['std']:.2f}"
                })
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)

            # Visualização
            st.markdown("### 📈 Visualization")
            fig = go.Figure()
            properties = list(predictions.keys())
            means = [predictions[p]['mean'] for p in properties]
            ci_lower = [predictions[p]['ci_lower'] for p in properties]
            ci_upper = [predictions[p]['ci_upper'] for p in properties]
            # Barras com erro
            fig.add_trace(go.Bar(
                x=properties,
                y=means,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[u - m for u, m in zip(ci_upper, means)],
                    arrayminus=[m - l for m, l in zip(means, ci_lower)]
                ),
                marker_color='#1f77b4',
                name='Predicted Value'
            ))
            fig.update_layout(
                title="Predicted Elastic Properties with 95% Confidence Intervals",
                xaxis_title="Property",
                yaxis_title="Value (GPa)",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Métricas destacadas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "E₁ (Fiber Direction)",
                    f"{predictions['E1']['mean']:.2f} GPa",
                    delta=f"±{predictions['E1']['std']:.2f}"
                )
            with col2:
                st.metric(
                    "E₂ (Transverse)",
                    f"{predictions['E2']['mean']:.2f} GPa",
                    delta=f"±{predictions['E2']['std']:.2f}"
                )
            with col3:
                anisotropy = predictions['E1']['mean'] / predictions['E2']['mean']
                st.metric(
                    "Anisotropy Index",
                    f"{anisotropy:.2f}",
                    delta="E₁/E₂"
                )

# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================

with tab3:
    st.header("📈 Model Performance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first")
    else:
        predictor = st.session_state.predictor
        st.markdown("### 🎯 Model Comparison")
        # Comparação real dos modelos treinados usando dados do usuário
        scenario_map = {
            'Full Measurements': 'full',
            'Partial - Longitudinal Only': 'partial_L',
            'Minimal Measurements': 'partial_minimal',
            'Minimal - Only V_L(Z) and V_T(Z)': 'minimal_z'
        }
        perf_rows = []
        input_data = st.session_state.input_data
        for scenario, feature_set in scenario_map.items():
            if feature_set in predictor.models:
                # Garantir que todas as features estejam presentes
                if feature_set == 'minimal_z':
                    feature_names = ['rho', 'thickness', 'temperature', 'V_L_Z', 'V_T_Z']
                else:
                    feature_names = predictor.prepare_features(st.session_state.training_data, feature_set).columns.tolist()
                input_data_filled = fill_features(input_data, feature_names)
                # Treinar novamente para obter os resultados de todos os modelos
                results, best_model = predictor.train_models(st.session_state.training_data, feature_set=feature_set)
                for model_name, res in results.items():
                    # Prever usando dados reais do usuário
                    try:
                        pred = predictor.models[feature_set].predict(
                            predictor.scalers[feature_set].transform(pd.DataFrame([input_data_filled]))
                        )[0]
                        pred_dict = {name: value for name, value in zip(predictor.target_names, pred)}
                    except Exception:
                        pred_dict = {name: np.nan for name in predictor.target_names}
                    perf_rows.append({
                        'Scenario': scenario,
                        'Model': model_name,
                        'R² Test': res['r2_test'],
                        'RMSE (GPa)': res['rmse_test'],
                        'CV Mean': res['cv_mean'],
                        'Best': '✓' if model_name == best_model else '',
                        **pred_dict
                    })
        df_perf = pd.DataFrame(perf_rows)
        st.dataframe(df_perf, use_container_width=True)
        # Destacar o melhor modelo de cada cenário
        st.markdown("### 🏆 Best Model per Scenario")
        for scenario in scenario_map:
            best = df_perf[(df_perf['Scenario'] == scenario) & (df_perf['Best'] == '✓')]
            if not best.empty:
                row = best.iloc[0]
                st.success(f"**{scenario}:** {row['Model']} (R² Test = {row['R² Test']:.3f}, RMSE = {row['RMSE (GPa)']:.3f} GPa)")

# ============================================================================
# TAB 4: FEATURE ANALYSIS
# ============================================================================

with tab4:
    st.header("🔍 Feature Importance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first")
    else:
        predictor = st.session_state.predictor
        # Seletor de modelo para análise de importância
        model_options = ['Random Forest', 'Gradient Boosting', 'Neural Network']
        selected_model = st.selectbox(
            "Select ML Model for Feature Importance",
            model_options,
            index=0,
            key='feature_importance_model'
        )
        # Forçar o modelo ativo para o selecionado
        feature_set = 'full'
        original_model = predictor.models[feature_set]
        predictor.models[feature_set] = predictor.train_models(
            st.session_state.training_data, feature_set=feature_set
        )[0][selected_model]['model']
        # Feature importance
        importance = predictor.feature_importance(feature_set=feature_set)
        # Restaurar o modelo original
        predictor.models[feature_set] = original_model
        if importance is not None:
            st.markdown("### 📊 Feature Importance Ranking")
            # Gráfico
            fig = px.bar(
                importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Which measurements matter most?",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            # Tabela
            st.dataframe(importance, use_container_width=True)
            # Insights
            st.markdown("### 💡 Key Findings")
            top_feature = importance.iloc[0]['Feature']
            top_importance = importance.iloc[0]['Importance']
            st.success(f"""
            **Most Important Feature**: {top_feature} ({top_importance:.1%} importance)
            This feature has the strongest influence on predicting elastic properties.
            Focus measurement efforts here for maximum accuracy.
            """)
            # Correlação entre features
            if st.session_state.training_data is not None:
                st.markdown("### 🔗 Feature Correlations")
                data = st.session_state.training_data
                feature_cols = ['V_L_0', 'V_L_90', 'V_L_Z', 'V_T_0', 'V_T_90', 'V_T_Z']
                corr_matrix = data[feature_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix of Velocity Measurements"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A importância das features só está disponível para o modelo Random Forest. "
                    "Selecione ou treine esse modelo para visualizar a análise de importância das medições.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 Ultrasonic Analysis + ML System | Developed with Streamlit</p>
    </div>
""", unsafe_allow_html=True)