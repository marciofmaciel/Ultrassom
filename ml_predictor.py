# ml_predictor.py
"""
Sistema de Machine Learning para Previsão de Propriedades Elásticas
Laminados de Fibra de Carbono
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

class ElasticPropertiesPredictor:
    """
    Preditor de propriedades elásticas usando Machine Learning
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.target_names = ['E1', 'E2', 'E3', 'G12', 'G13', 'G23']
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Gera dados sintéticos realistas para treinamento
        Baseado em relações físicas conhecidas de laminados de fibra de carbono
        """
        np.random.seed(42)
        
        # Parâmetros típicos de fibra de carbono/epóxi
        # Fração volumétrica de fibras (Vf)
        Vf = np.random.uniform(0.50, 0.70, n_samples)
        
        # Densidade (função de Vf)
        rho_fiber = 1800  # kg/m³ (fibra de carbono)
        rho_matrix = 1200  # kg/m³ (epóxi)
        rho = Vf * rho_fiber + (1 - Vf) * rho_matrix
        
        # Espessura
        thickness = np.random.uniform(2.0, 10.0, n_samples)  # mm
        
        # Temperatura (afeta propriedades da matriz)
        temperature = np.random.uniform(15, 35, n_samples)  # °C
        temp_factor = 1 - 0.0003 * (temperature - 20)
        
        # Propriedades das fibras e matriz
        Ef = 230e9  # Pa - módulo da fibra
        Em = 3.5e9  # Pa - módulo da matriz
        Gf = 15e9   # Pa - cisalhamento fibra
        Gm = 1.3e9  # Pa - cisalhamento matriz
        
        # Regra das misturas (aproximação)
        E1_base = Vf * Ef + (1 - Vf) * Em  # Paralelo às fibras
        E2_base = (Em * Ef) / (Vf * Em + (1 - Vf) * Ef)  # Perpendicular
        E3_base = E2_base * 0.9  # Através da espessura (um pouco menor)
        
        G12_base = (Gm * Gf) / (Vf * Gm + (1 - Vf) * Gf)
        G13_base = G12_base * 0.95
        G23_base = G12_base * 0.85
        
        # Aplicar fator de temperatura
        E1 = E1_base * temp_factor / 1e9  # Converter para GPa
        E2 = E2_base * temp_factor / 1e9
        E3 = E3_base * temp_factor / 1e9
        G12 = G12_base * temp_factor / 1e9
        G13 = G13_base * temp_factor / 1e9
        G23 = G23_base * temp_factor / 1e9
        
        # Calcular velocidades a partir das propriedades
        # V_L = sqrt(C_ii / rho), onde C_ii ≈ E (aproximação)
        V_L_0 = np.sqrt(E1 * 1e9 / rho)  # m/s
        V_L_90 = np.sqrt(E2 * 1e9 / rho)
        V_L_Z = np.sqrt(E3 * 1e9 / rho)
        
        # V_T = sqrt(G / rho)
        V_T_0 = np.sqrt(G12 * 1e9 / rho)
        V_T_90 = np.sqrt(G23 * 1e9 / rho)
        V_T_Z = np.sqrt(G13 * 1e9 / rho)
        
        # Adicionar ruído realista (variabilidade experimental)
        noise_level = 0.02  # 2% de variação
        V_L_0 *= np.random.normal(1, noise_level, n_samples)
        V_L_90 *= np.random.normal(1, noise_level, n_samples)
        V_L_Z *= np.random.normal(1, noise_level, n_samples)
        V_T_0 *= np.random.normal(1, noise_level, n_samples)
        V_T_90 *= np.random.normal(1, noise_level, n_samples)
        V_T_Z *= np.random.normal(1, noise_level, n_samples)
        
        # Criar DataFrame
        data = pd.DataFrame({
            'Vf': Vf,
            'rho': rho,
            'thickness': thickness,
            'temperature': temperature,
            'V_L_0': V_L_0,
            'V_L_90': V_L_90,
            'V_L_Z': V_L_Z,
            'V_T_0': V_T_0,
            'V_T_90': V_T_90,
            'V_T_Z': V_T_Z,
            'E1': E1,
            'E2': E2,
            'E3': E3,
            'G12': G12,
            'G13': G13,
            'G23': G23
        })
        
        return data
    
    def prepare_features(self, data, feature_set='full'):
        """
        Prepara features para treinamento
        
        feature_set:
        - 'full': todas as velocidades
        - 'partial_L': apenas velocidades longitudinais
        - 'partial_minimal': apenas V_L_0 e V_L_90
        """
        if feature_set == 'full':
            features = ['rho', 'thickness', 'temperature', 
                       'V_L_0', 'V_L_90', 'V_L_Z', 
                       'V_T_0', 'V_T_90', 'V_T_Z']
        elif feature_set == 'partial_L':
            features = ['rho', 'thickness', 'temperature',
                       'V_L_0', 'V_L_90', 'V_L_Z']
        elif feature_set == 'partial_minimal':
            features = ['rho', 'thickness', 'temperature',
                       'V_L_0', 'V_L_90']
        elif feature_set == 'minimal_z':
            features = ['rho', 'thickness', 'temperature', 'V_L_Z', 'V_T_Z']
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        self.feature_names = features
        return data[features]
    
    def train_models(self, data, feature_set='full', test_size=0.2):
        """
        Treina múltiplos modelos e seleciona o melhor
        """
        print(f"\n{'='*70}")
        print(f"TREINAMENTO DE MODELOS ML - Feature Set: {feature_set}")
        print(f"{'='*70}\n")
        
        # Preparar dados
        X = self.prepare_features(data, feature_set)
        y = data[self.target_names]
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Normalizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[feature_set] = scaler
        
        # Definir modelos a testar
        models_to_test = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )),
            'Neural Network': MultiOutputRegressor(MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            ))
        }
        
        # Treinar e avaliar cada modelo
        results = {}
        best_score = -np.inf
        best_model_name = None
        
        for name, model in models_to_test.items():
            print(f"Treinando {name}...")
            
            # Treinar
            model.fit(X_train_scaled, y_train)
            
            # Prever
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Avaliar
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=5, scoring='r2', n_jobs=-1
            )
            
            results[name] = {
                'model': model,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_test': rmse_test,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  R² Train: {r2_train:.4f}")
            print(f"  R² Test:  {r2_test:.4f}")
            print(f"  RMSE:     {rmse_test:.4f} GPa")
            print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print()
            
            # Atualizar melhor modelo
            if r2_test > best_score:
                best_score = r2_test
                best_model_name = name
        
        # Salvar melhor modelo
        best_model = results[best_model_name]['model']
        self.models[feature_set] = best_model
        
        print(f"✓ Melhor modelo: {best_model_name} (R² = {best_score:.4f})")
        print(f"{'='*70}\n")
        
        self.is_trained = True
        return results, best_model_name
    
    def predict(self, input_data, feature_set='full'):
        """
        Faz previsão de propriedades elásticas
        
        input_data: dict com features
        Exemplo: {'rho': 1573, 'thickness': 4.87, 'temperature': 21,
                  'V_L_0': 3498, 'V_L_90': 2001}
        """
        if not self.is_trained:
            raise ValueError("Modelo não treinado! Execute train_models() primeiro.")
        
        if feature_set not in self.models:
            raise ValueError(f"Feature set '{feature_set}' não disponível.")

        # Checar se todas as features estão presentes no input ANTES de criar o DataFrame
        missing = [k for k in self.feature_names if k not in input_data]
        if missing:
            raise ValueError(
                f"As seguintes features estão faltando no input: {missing}.\n"
                f"O modelo foi treinado com as features: {self.feature_names}.\n"
                "Forneça todas as features esperadas no dicionário de entrada para predict. "
                "Se não tiver o valor, coloque np.nan ou 0."
            )

        # Preencher automaticamente features ausentes com np.nan (garantia extra, mas não deve ocorrer)
        input_filled = {k: input_data.get(k, np.nan) for k in self.feature_names}
        # Garantir ordem e apenas as features corretas
        input_df = pd.DataFrame([[input_filled[k] for k in self.feature_names]], columns=self.feature_names)
        X = input_df

        # Normalizar
        X_scaled = self.scalers[feature_set].transform(X)

        # Prever
        predictions = self.models[feature_set].predict(X_scaled)[0]

        # Criar dicionário de resultados
        results = {
            name: value 
            for name, value in zip(self.target_names, predictions)
        }

        return results
    
    def predict_with_confidence(self, input_data, feature_set='full', n_bootstrap=100):
        """
        Previsão com intervalo de confiança usando bootstrap
        """
        if feature_set not in self.models:
            raise ValueError(f"Feature set '{feature_set}' não disponível.")

        model = self.models[feature_set]
        # Garantir que só as features corretas estejam no DataFrame
        input_df = pd.DataFrame([input_data])
        X = input_df[[f for f in self.feature_names if f in input_df.columns]]

        # Se for Random Forest, usar previsões das árvores individuais
        if isinstance(model, RandomForestRegressor):
            X_scaled = self.scalers[feature_set].transform(X)
            # Previsões de cada árvore
            tree_predictions = np.array([
                tree.predict(X_scaled)[0]
                for tree in model.estimators_
            ])
            # Estatísticas
            mean_pred = tree_predictions.mean(axis=0)
            std_pred = tree_predictions.std(axis=0)
            ci_lower = np.percentile(tree_predictions, 2.5, axis=0)
            ci_upper = np.percentile(tree_predictions, 97.5, axis=0)
            results = {}
            for i, name in enumerate(self.target_names):
                results[name] = {
                    'mean': mean_pred[i],
                    'std': std_pred[i],
                    'ci_lower': ci_lower[i],
                    'ci_upper': ci_upper[i]
                }
            return results
        else:
            # Para outros modelos, retornar apenas previsão pontual
            predictions = self.predict(input_data, feature_set)
            return {
                name: {'mean': value, 'std': 0, 'ci_lower': value, 'ci_upper': value}
                for name, value in predictions.items()
            }
    
    def feature_importance(self, feature_set='full'):
        """
        Retorna importância das features (para Random Forest)
        """
        if feature_set not in self.models:
            raise ValueError(f"Feature set '{feature_set}' não disponível.")
        
        model = self.models[feature_set]
        
        if isinstance(model, RandomForestRegressor):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return feature_importance_df
        else:
            return None
    
    def save_models(self, filepath='ml_models.pkl'):
        """
        Salva modelos treinados
        """
        data_to_save = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"✓ Modelos salvos em: {filepath}")
    
    def load_models(self, filepath='ml_models.pkl'):
        """
        Carrega modelos treinados
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.scalers = data['scalers']
        self.feature_names = data['feature_names']
        self.target_names = data['target_names']
        self.is_trained = data['is_trained']
        
        print(f"✓ Modelos carregados de: {filepath}")


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SISTEMA DE MACHINE LEARNING - PREVISÃO DE PROPRIEDADES ELÁSTICAS")
    print("="*70 + "\n")
    
    # Criar preditor
    predictor = ElasticPropertiesPredictor()
    
    # Gerar dados sintéticos
    print("Gerando dados sintéticos de treinamento...")
    data = predictor.generate_synthetic_data(n_samples=1000)
    print(f"✓ {len(data)} amostras geradas\n")
    
    # Treinar com diferentes conjuntos de features
    print("\n" + "="*70)
    print("CENÁRIO 1: Todas as medições disponíveis")
    print("="*70)
    results_full, best_full = predictor.train_models(data, feature_set='full')
    
    print("\n" + "="*70)
    print("CENÁRIO 2: Apenas velocidades longitudinais")
    print("="*70)
    results_partial_L, best_partial_L = predictor.train_models(data, feature_set='partial_L')
    
    print("\n" + "="*70)
    print("CENÁRIO 3: Medições mínimas (apenas V_L_0 e V_L_90)")
    print("="*70)
    results_minimal, best_minimal = predictor.train_models(data, feature_set='partial_minimal')
    
    # Teste de previsão
    print("\n" + "="*70)
    print("TESTE DE PREVISÃO")
    print("="*70 + "\n")
    
    # Caso de teste realista (valores de exemplo)
    base_input = {
        'rho': 1573,
        'thickness': 4.87,
        'temperature': 21,
        'V_L_0': 3498,
        'V_L_90': 2001,
        'V_L_Z': 3001,
        'V_T_0': 1847,
        'V_T_90': 1423,
        'V_T_Z': 1589
    }

    # Função para garantir que todas as features estejam presentes
    def fill_features(input_dict, feature_names):
        filled = {k: input_dict.get(k, np.nan) for k in feature_names}
        missing = [k for k in feature_names if k not in input_dict]
        if missing:
            print(f"[AVISO] As seguintes features estavam faltando e foram preenchidas com np.nan: {missing}")
        return filled

    # Teste para cada feature_set
    for feature_set in ['full', 'partial_minimal']:
        feature_names = predictor.prepare_features(data, feature_set).columns.tolist()
        test_input = fill_features(base_input, feature_names)
        print(f"\nInput de teste para feature_set='{feature_set}':")
        for key, value in test_input.items():
            print(f"  {key}: {value}")
        print()

        print(f"Previsão (feature_set='{feature_set}'):")
        predictions = predictor.predict(test_input, feature_set=feature_set)
        for prop, value in predictions.items():
            print(f"  {prop}: {value:.2f} GPa")
        print()

    # Previsão com intervalo de confiança para 'full'
    feature_names_full = predictor.prepare_features(data, 'full').columns.tolist()
    test_input_full = fill_features(base_input, feature_names_full)
    print("Previsão com intervalo de confiança (95%):")
    predictions_ci = predictor.predict_with_confidence(test_input_full, feature_set='full')
    for prop, stats in predictions_ci.items():
        print(f"  {prop}: {stats['mean']:.2f} GPa "
              f"[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]")
    print()
    
    # Feature importance
    print("Importância das Features:")
    importance = predictor.feature_importance(feature_set='full')
    if importance is not None:
        print(importance.to_string(index=False))
    else:
        print("Importância das features não disponível para o modelo selecionado.")
    print()
    
    # Salvar modelos
    predictor.save_models('elastic_ml_models.pkl')
    
    print("\n✓ Sistema ML configurado com sucesso!\n")