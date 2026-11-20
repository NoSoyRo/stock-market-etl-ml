"""
Fitness Evaluator - Evalúa la calidad de un modelo (cromosoma)
"""
import numpy as np
from typing import Dict, Any
from loguru import logger

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    logger.warning("⚠️  scikit-learn no instalado")


class FitnessEvaluator:
    """
    Evalúa el fitness de un modelo LSTM
    
    Fitness puede ser:
    - Simple: val_rmse (minimizar)
    - Multiobjetivo: combinación de métricas
    """
    
    def __init__(
        self,
        metric: str = 'val_rmse',
        weights: Dict[str, float] = None
    ):
        """
        Args:
            metric: Métrica principal ('val_rmse', 'val_mae', 'composite')
            weights: Pesos para fitness multiobjetivo
        """
        self.metric = metric
        self.weights = weights or {
            'val_rmse': 0.4,
            'val_mae': 0.3,
            'sharpe_ratio': 0.2,
            'win_rate': 0.1
        }
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray = None
    ) -> float:
        """
        Calcular fitness (menor es mejor para GA)
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_train: Valores de entrenamiento (para Sharpe)
            
        Returns:
            Fitness score (menor es mejor)
        """
        metrics = self.calculate_all_metrics(y_true, y_pred, y_train)
        
        if self.metric == 'val_rmse':
            return metrics['rmse']
        
        elif self.metric == 'val_mae':
            return metrics['mae']
        
        elif self.metric == 'composite':
            # Fitness multiobjetivo
            fitness = 0.0
            
            # RMSE (minimizar)
            if 'val_rmse' in self.weights:
                fitness += self.weights['val_rmse'] * metrics['rmse']
            
            # MAE (minimizar)
            if 'val_mae' in self.weights:
                fitness += self.weights['val_mae'] * metrics['mae']
            
            # Sharpe ratio (maximizar → invertir)
            if 'sharpe_ratio' in self.weights and 'sharpe_ratio' in metrics:
                # Invertir para minimizar (1 / sharpe)
                sharpe = max(metrics['sharpe_ratio'], 0.01)  # Evitar división por 0
                fitness += self.weights['sharpe_ratio'] * (1 / sharpe)
            
            # Win rate (maximizar → invertir)
            if 'win_rate' in self.weights and 'win_rate' in metrics:
                fitness += self.weights['win_rate'] * (1 - metrics['win_rate'])
            
            return fitness
        
        else:
            raise ValueError(f"Métrica desconocida: {self.metric}")
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calcular todas las métricas disponibles
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_train: Valores de entrenamiento (opcional)
            
        Returns:
            Dict con métricas
        """
        metrics = {}
        
        # RMSE
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAE
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # R²
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE
        mask = y_true != 0
        if mask.any():
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.inf
        
        # Win Rate (% predicciones correctas de dirección)
        if len(y_true) > 1:
            # Calcular dirección real
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            
            metrics['win_rate'] = np.mean(true_direction == pred_direction)
        
        # Sharpe Ratio (si tenemos datos de training).
        # NOTA (auditoría Handoff 01): en TrainingPipeline._preprocess los targets
        # y_train vienen escalados con MinMaxScaler en [0, 1], por lo que
        # np.diff(y_train)/y_train[:-1] NO es un retorno financiero real (y OptimizationPipeline
        # usa val_rmse directamente sin pasar por FitnessEvaluator.evaluate). Se conserva
        # sin cambios hasta decidir si se elimina del fitness compuesto o se pasan retornos reales.
        if y_train is not None and len(y_train) > 1:
            returns = np.diff(y_train) / y_train[:-1]
            if returns.std() > 0:
                metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
            else:
                metrics['sharpe_ratio'] = 0.0
        
        return metrics
    
    def is_better(self, fitness1: float, fitness2: float) -> bool:
        """
        Determinar si fitness1 es mejor que fitness2
        
        Args:
            fitness1: Primer fitness
            fitness2: Segundo fitness
            
        Returns:
            True si fitness1 es mejor (menor)
        """
        return fitness1 < fitness2
