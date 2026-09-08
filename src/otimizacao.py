import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

def otimizar_recursos_colonia():
    """Simula a previsão e otimização de recursos da colônia usando Gradiente Descendente e MSE."""
    print("\n" + "=" * 60)
    print("      NCAS - MÓDULO DE OTIMIZAÇÃO PREDITIVA (MACHINE LEARNING)")
    print("=" * 60)
    print("Iniciando treinamento do modelo preditivo para consumo de energia/oxigênio...")
    
    # Dados históricos simulados (ex: temperatura externa vs consumo de energia)
    X_treino = np.array([[10], [15], [20], [25], [30]]) 
    y_treino = np.array([50, 70, 90, 110, 130]) 
    
    # Instanciando o SGDRegressor (Gradiente Descendente Estocástico) com regularização L2 (Ridge)
    modelo_sgd = SGDRegressor(
        loss="squared_error", 
        penalty="l2", 
        alpha=0.0001, 
        max_iter=1000, 
        random_state=42
    )
    
    # Treinando o modelo
    modelo_sgd.fit(X_treino, y_treino)
    
    # Previsão e cálculo do Erro Quadrático Médio (MSE)
    y_previsto = modelo_sgd.predict(X_treino)
    mse = mean_squared_error(y_treino, y_previsto)
    
    print("\n[SUCESSO] Modelo otimizado com Gradiente Descendente!")
    print(f"-> Erro Quadrático Médio (MSE) alcançado: {mse:.4f}")
    print("-> Regularização L2 aplicada para controlar complexidade e evitar overfitting.")
    print("=" * 60)