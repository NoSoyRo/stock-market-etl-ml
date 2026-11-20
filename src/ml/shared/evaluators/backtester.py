"""
Backtester - Walk-forward validation para evaluar modelos en producción simulada
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from .metrics import Metrics


class Backtester:
    """
    Backtester con walk-forward validation
    
    Simula cómo se habría comportado el modelo en producción:
    1. Entrenar con datos hasta fecha T
    2. Predecir T+1, T+2, ..., T+N
    3. Avanzar ventana temporal
    4. Repetir
    
    Esto evita data leakage y simula realidad de predicción
    """
    
    def __init__(
        self,
        initial_train_days: int = 252,  # 1 año
        retrain_frequency: int = 21,    # Reentrenar cada mes
        test_days: int = 5,             # Predecir 5 días adelante
        commission_pct: float = Metrics.DEFAULT_COMMISSION_PCT,
        slippage_pct: float = Metrics.DEFAULT_SLIPPAGE_PCT,
    ):
        """
        Args:
            initial_train_days: Días iniciales de entrenamiento
            retrain_frequency: Cada cuántos días reentrenar
            test_days: Días a predecir en cada ventana
            commission_pct: Comisión por lado (default 0.25% BMV)
            slippage_pct: Slippage por lado (default 0.05%)
        """
        self.initial_train_days = initial_train_days
        self.retrain_frequency = retrain_frequency
        self.test_days = test_days
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        self.results: List[Dict] = []
        self.all_predictions = []
        self.all_actuals = []
        self.trade_returns_gross: List[float] = []
        self.trade_returns_net: List[float] = []
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        model_trainer,
        date_column: str = 'fecha',
        target_column: str = 'close'
    ) -> Dict:
        """
        Ejecutar backtesting completo
        
        Args:
            data: DataFrame con datos históricos
            model_trainer: Objeto con método train() y predict()
            date_column: Nombre de columna de fecha
            target_column: Nombre de columna objetivo
            
        Returns:
            Dict con métricas agregadas
        """
        logger.info("🔄 INICIANDO BACKTESTING")
        logger.info("=" * 80)
        logger.info(f"Datos totales: {len(data)} registros")
        logger.info(f"Train inicial: {self.initial_train_days} días")
        logger.info(f"Reentrenar cada: {self.retrain_frequency} días")
        logger.info(f"Predecir: {self.test_days} días adelante")
        logger.info(
            f"Costos BMV: comisión={self.commission_pct*100:.2f}%/lado, "
            f"slippage={self.slippage_pct*100:.3f}%/lado, "
            f"round-trip≈{Metrics.round_trip_cost(self.commission_pct, self.slippage_pct)*100:.2f}%"
        )
        logger.info("")
        
        # Ordenar por fecha
        data = data.sort_values(date_column).reset_index(drop=True)
        
        # Resetear resultados
        self.results = []
        self.all_predictions = []
        self.all_actuals = []
        self.trade_returns_gross = []
        self.trade_returns_net = []
        
        # Variables de control
        current_pos = self.initial_train_days
        step = 0
        
        while current_pos + self.test_days < len(data):
            step += 1
            
            # Ventana de entrenamiento
            train_data = data.iloc[:current_pos]
            
            # Ventana de test
            test_data = data.iloc[current_pos:current_pos + self.test_days]
            
            logger.info(f"\n📊 Ventana {step}")
            logger.info(f"   Train: {train_data[date_column].iloc[0]} a {train_data[date_column].iloc[-1]}")
            logger.info(f"   Test:  {test_data[date_column].iloc[0]} a {test_data[date_column].iloc[-1]}")
            
            # Entrenar modelo
            try:
                model_trainer.train(train_data)
                
                # Hacer predicciones
                predictions = model_trainer.predict(test_data)
                actuals = test_data[target_column].values
                
                # Guardar predicciones
                self.all_predictions.extend(predictions)
                self.all_actuals.extend(actuals)

                # Simular trades direccionales long-only con costos BMV
                gross, net = self._simulate_directional_returns(actuals, predictions)
                self.trade_returns_gross.extend(gross.tolist())
                self.trade_returns_net.extend(net.tolist())
                
                # Calcular métricas para esta ventana
                window_metrics = Metrics.calculate_all_metrics(
                    y_true=actuals,
                    y_pred=predictions
                )
                
                # Guardar resultado de ventana
                window_result = {
                    'step': step,
                    'train_start': train_data[date_column].iloc[0],
                    'train_end': train_data[date_column].iloc[-1],
                    'test_start': test_data[date_column].iloc[0],
                    'test_end': test_data[date_column].iloc[-1],
                    'predictions': predictions,
                    'actuals': actuals,
                    **window_metrics
                }
                
                self.results.append(window_result)
                
                logger.info(f"   ✅ RMSE: {window_metrics['rmse']:.4f}, Win Rate: {window_metrics['win_rate']*100:.1f}%")
                
            except Exception as e:
                logger.error(f"   ❌ Error en ventana {step}: {e}")
            
            # Avanzar ventana
            current_pos += self.retrain_frequency
        
        # Calcular métricas agregadas
        logger.info("\n" + "=" * 80)
        logger.info("📊 RESULTADOS FINALES DEL BACKTESTING")
        logger.info("=" * 80)
        
        aggregate_metrics = self._calculate_aggregate_metrics()
        
        Metrics.print_metrics(aggregate_metrics)
        
        return aggregate_metrics
    
    def _calculate_aggregate_metrics(self) -> Dict:
        """
        Calcular métricas agregadas de todo el backtest
        
        Returns:
            Dict con métricas
        """
        if not self.all_predictions:
            return {}
        
        # Convertir a arrays
        all_preds = np.array(self.all_predictions)
        all_actuals = np.array(self.all_actuals)
        
        # Calcular métricas globales (regresión + dirección)
        aggregate = Metrics.calculate_all_metrics(
            y_true=all_actuals,
            y_pred=all_preds
        )
        
        # Métricas duales de portafolio con costos BMV (KB-001 / KB-003).
        # _simulate_directional_returns produce retornos diarios consecutivos
        # (np.diff(y_true)/y_true[:-1] sobre barras diarias de precio), por lo que
        # periods_per_year=252 es la anualización correcta.
        gross, net = self.get_trade_returns()
        if len(gross) > 0:
            dual = Metrics.calculate_dual_metrics(gross, net, periods_per_year=252)
            aggregate.update(dual)
            aggregate['sharpe_ratio'] = dual['sharpe_gross']
            aggregate['win_rate_net'] = dual['win_rate_net']

        # Agregar info de ventanas
        aggregate['num_windows'] = len(self.results)
        aggregate['total_predictions'] = len(self.all_predictions)
        aggregate['commission_pct'] = self.commission_pct
        aggregate['slippage_pct'] = self.slippage_pct
        aggregate['round_trip_cost'] = Metrics.round_trip_cost(
            self.commission_pct, self.slippage_pct
        )
        
        # Calcular estabilidad (std de métricas entre ventanas)
        if len(self.results) > 1:
            window_rmse = [r['rmse'] for r in self.results]
            aggregate['rmse_stability'] = np.std(window_rmse)
        
        return aggregate

    def _simulate_directional_returns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula trades long-only: entra si predicción sube, aplica costos round-trip.

        Returns:
            (returns_gross, returns_net) por período con señal activa
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) < 2:
            return np.array([]), np.array([])

        pred_up = np.diff(y_pred) > 0
        true_rets = np.diff(y_true) / y_true[:-1]

        gross = np.where(pred_up, true_rets, 0.0)
        net = Metrics.apply_costs_to_returns(
            gross,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            trade_mask=pred_up,
        )
        return gross, net

    def get_trade_returns(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retornos brutos y netos de trades simulados."""
        return (
            np.array(self.trade_returns_gross, dtype=float),
            np.array(self.trade_returns_net, dtype=float),
        )
    
    def get_predictions_df(self) -> pd.DataFrame:
        """
        Obtener DataFrame con todas las predicciones
        
        Returns:
            DataFrame con fecha, actual, predicted
        """
        records = []
        
        for result in self.results:
            for i, (actual, pred) in enumerate(zip(result['actuals'], result['predictions'])):
                records.append({
                    'date': result['test_start'] + timedelta(days=i),
                    'actual': actual,
                    'predicted': pred,
                    'error': actual - pred,
                    'error_pct': (actual - pred) / actual * 100,
                    'window': result['step']
                })
        
        return pd.DataFrame(records)
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Graficar resultados del backtesting
        
        Args:
            save_path: Ruta donde guardar gráfico
        """
        try:
            import matplotlib.pyplot as plt
            
            df = self.get_predictions_df()
            
            if df.empty:
                logger.warning("⚠️  No hay datos para graficar")
                return
            
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Predicciones vs Actual
            ax1 = axes[0]
            ax1.plot(df.index, df['actual'], 'b-', label='Actual', linewidth=2)
            ax1.plot(df.index, df['predicted'], 'r--', label='Predicción', linewidth=1.5)
            ax1.set_title('Backtesting: Predicciones vs Valores Reales')
            ax1.set_xlabel('Índice')
            ax1.set_ylabel('Precio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Error
            ax2 = axes[1]
            ax2.plot(df.index, df['error_pct'], 'g-', linewidth=1)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_title('Error de Predicción (%)')
            ax2.set_xlabel('Índice')
            ax2.set_ylabel('Error %')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Gráfico guardado en {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("⚠️  matplotlib no disponible para graficar")
