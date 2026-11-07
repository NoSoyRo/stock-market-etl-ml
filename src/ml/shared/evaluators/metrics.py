"""
Metrics - Suite completa de métricas para evaluación de modelos
"""
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger

# BMV defaults (KB-003) — ~0.55% round-trip per bmv_market_rules.yaml
DEFAULT_COMMISSION_PCT = 0.0025
DEFAULT_SLIPPAGE_PCT = 0.0005


class Metrics:
    """
    Suite completa de métricas para modelos de predicción
    
    Métricas implementadas:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - R²: Coeficiente de determinación
    - Win Rate: Proporción de direcciones correctas
    - Sharpe Ratio: Retorno ajustado por riesgo
    - Max Drawdown: Máxima caída desde pico
    - Calmar Ratio: Retorno anualizado / Max Drawdown
    - Sortino Ratio: Sharpe pero solo con downside volatility
    """

    DEFAULT_COMMISSION_PCT = DEFAULT_COMMISSION_PCT
    DEFAULT_SLIPPAGE_PCT = DEFAULT_SLIPPAGE_PCT

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error
        
        RMSE penaliza más los errores grandes
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error
        
        MAE trata todos los errores por igual
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error
        
        MAPE expresa error como porcentaje
        """
        # Evitar división por cero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R² - Coeficiente de determinación
        
        R² = 1 indica predicción perfecta
        R² = 0 indica predicción igual a la media
        R² < 0 indica predicción peor que la media
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Accuracy de la dirección del cambio
        
        ¿Predijimos correctamente si sube o baja?
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calcular cambios
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Comparar direcciones (mismo signo = correcto)
        correct = np.sum(np.sign(true_changes) == np.sign(pred_changes))
        
        return correct / len(true_changes)
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252.0,
    ) -> float:
        """
        Sharpe Ratio - Retorno ajustado por riesgo
        
        Sharpe = (Retorno promedio - Risk free rate) / Volatilidad
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo anualizada (default 0%)
            periods_per_year: Periodos por año (252 diario; 50.4 si el periodo es de 5 días)
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Convertir risk_free_rate anual a tasa por periodo
        period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        excess_returns = returns - period_rf
        
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
    
    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252.0,
    ) -> float:
        """
        Sortino Ratio - Similar a Sharpe pero solo penaliza downside
        
        Sortino = (Retorno promedio - Risk free rate) / Downside Deviation
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0
        
        # Convertir risk_free_rate anual a tasa por periodo
        period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        excess_returns = returns - period_rf
        
        # Solo retornos negativos para downside deviation
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        
        return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    
    @staticmethod
    def max_drawdown(cumulative_returns: np.ndarray) -> float:
        """
        Maximum Drawdown - Máxima caída desde pico

        Args:
            cumulative_returns: Curva de capital (índice de riqueza, ej. cumprod(1+r))
        """
        if len(cumulative_returns) == 0:
            return 0.0

        wealth = np.asarray(cumulative_returns, dtype=float)
        if np.all(wealth <= 0):
            return -1.0

        running_max = np.maximum.accumulate(wealth)
        drawdown = (wealth - running_max) / running_max

        return float(np.min(drawdown))
    
    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        cumulative_returns: np.ndarray,
        periods_per_year: float = 252.0,
    ) -> float:
        """
        Calmar Ratio - Retorno anualizado / Max Drawdown
        
        Mide retorno ajustado por máxima caída
        """
        annual_return = np.mean(returns) * periods_per_year
        mdd = abs(Metrics.max_drawdown(cumulative_returns))
        
        if mdd == 0:
            return 0.0
        
        return annual_return / mdd
    
    @staticmethod
    def round_trip_cost(
        commission_pct: float = DEFAULT_COMMISSION_PCT,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    ) -> float:
        """Costo round-trip: entrada + salida (comisión + slippage por lado)."""
        return 2 * (commission_pct + slippage_pct)

    @staticmethod
    def net_return(
        gross_return: float,
        commission_pct: float = DEFAULT_COMMISSION_PCT,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    ) -> float:
        """Retorno neto tras restar costos de transacción BMV."""
        return gross_return - Metrics.round_trip_cost(commission_pct, slippage_pct)

    @staticmethod
    def apply_costs_to_returns(
        gross_returns: np.ndarray,
        commission_pct: float = DEFAULT_COMMISSION_PCT,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
        trade_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Resta costos round-trip en cada trade simulado.

        Args:
            gross_returns: Retornos brutos por período
            trade_mask: True donde hubo trade; default = posiciones activas (≠ 0)
        """
        gross_returns = np.asarray(gross_returns, dtype=float)
        if trade_mask is None:
            trade_mask = gross_returns != 0
        else:
            trade_mask = np.asarray(trade_mask, dtype=bool)

        net = gross_returns.copy()
        cost = Metrics.round_trip_cost(commission_pct, slippage_pct)
        net[trade_mask] -= cost
        return net

    @staticmethod
    def win_rate_from_returns(returns: np.ndarray) -> float:
        """Win rate neto: proporción de períodos con retorno positivo."""
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0
        return float(np.mean(returns > 0))

    @staticmethod
    def total_return(returns: np.ndarray) -> float:
        """Retorno total compuesto de una serie de retornos periódicos."""
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0
        return float(np.prod(1 + returns) - 1)

    @staticmethod
    def portfolio_metrics_from_returns(
        returns: np.ndarray,
        periods_per_year: float = 252,
    ) -> Dict[str, float]:
        """
        Métricas de portafolio a partir de retornos periódicos.

        Returns:
            sharpe_ratio, max_drawdown, calmar_ratio, win_rate, total_return
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "total_return": 0.0,
            }

        wealth = np.cumprod(1 + returns)
        return {
            "sharpe_ratio": Metrics.sharpe_ratio(returns, periods_per_year=periods_per_year),
            "max_drawdown": Metrics.max_drawdown(wealth),
            "calmar_ratio": Metrics.calmar_ratio(returns, wealth, periods_per_year=periods_per_year),
            "win_rate": Metrics.win_rate_from_returns(returns),
            "total_return": Metrics.total_return(returns),
        }

    @staticmethod
    def calculate_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
        """AUC-ROC si hay variación en labels; None si no aplica."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            return None
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            return None

    @staticmethod
    def calculate_dual_metrics(
        returns_gross: np.ndarray,
        returns_net: np.ndarray,
        periods_per_year: float = 252,
        y_true_binary: Optional[np.ndarray] = None,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Métricas duales KB-001: AUC + Sharpe bruto/neto + MDD + Calmar + win_rate neto.
        """
        gross = Metrics.portfolio_metrics_from_returns(returns_gross, periods_per_year)
        net = Metrics.portfolio_metrics_from_returns(returns_net, periods_per_year)

        dual: Dict[str, float] = {
            "sharpe_gross": gross["sharpe_ratio"],
            "sharpe_net": net["sharpe_ratio"],
            "max_drawdown": net["max_drawdown"],
            "calmar_ratio": net["calmar_ratio"],
            "win_rate_net": net["win_rate"],
            "total_return_gross": gross["total_return"],
            "total_return_net": net["total_return"],
        }

        auc = None
        if y_true_binary is not None and y_prob is not None:
            auc = Metrics.calculate_auc(y_true_binary, y_prob)
        if auc is not None:
            dual["auc"] = auc

        return dual

    @staticmethod
    def win_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Win Rate - Proporción de predicciones direccionales correctas
        
        Similar a direction_accuracy pero más simple
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calcular retornos
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Comparar signos
        correct = np.sum(np.sign(true_returns) == np.sign(pred_returns))
        
        return correct / len(true_returns)
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252.0,
    ) -> Dict[str, float]:
        """
        Calcular todas las métricas de una vez
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            risk_free_rate: Tasa libre de riesgo anualizada
            periods_per_year: Periodos por año para anualizar Sharpe/Sortino/Calmar (default 252)
            
        Returns:
            Dict con todas las métricas
        """
        # Convertir a numpy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Métricas básicas
        metrics = {
            'rmse': Metrics.rmse(y_true, y_pred),
            'mae': Metrics.mae(y_true, y_pred),
            'mape': Metrics.mape(y_true, y_pred),
            'r2': Metrics.r2_score(y_true, y_pred),
            'direction_accuracy': Metrics.direction_accuracy(y_true, y_pred),
            'win_rate': Metrics.win_rate(y_true, y_pred)
        }
        
        # Métricas basadas en retornos
        if len(y_true) >= 2:
            returns = np.diff(y_true) / y_true[:-1]
            cumulative_returns = np.cumprod(1 + returns) - 1
            
            metrics.update({
                'sharpe_ratio': Metrics.sharpe_ratio(
                    returns, risk_free_rate, periods_per_year=periods_per_year
                ),
                'sortino_ratio': Metrics.sortino_ratio(
                    returns, risk_free_rate, periods_per_year=periods_per_year
                ),
                'max_drawdown': Metrics.max_drawdown(cumulative_returns),
                'calmar_ratio': Metrics.calmar_ratio(
                    returns, cumulative_returns, periods_per_year=periods_per_year
                )
            })
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """
        Imprimir métricas en formato legible
        
        Args:
            metrics: Dict con métricas
        """
        logger.info("📊 MÉTRICAS DE EVALUACIÓN")
        logger.info("=" * 50)
        
        # Agrupar métricas
        error_metrics = ['rmse', 'mae', 'mape']
        quality_metrics = ['r2', 'direction_accuracy', 'win_rate']
        risk_metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio']
        
        logger.info("📉 Métricas de Error:")
        for metric in error_metrics:
            if metric in metrics:
                logger.info(f"   {metric.upper()}: {metrics[metric]:.6f}")
        
        logger.info("\n📈 Métricas de Calidad:")
        for metric in quality_metrics:
            if metric in metrics:
                if metric in ['direction_accuracy', 'win_rate']:
                    logger.info(f"   {metric.replace('_', ' ').title()}: {metrics[metric]*100:.2f}%")
                else:
                    logger.info(f"   {metric.upper()}: {metrics[metric]:.6f}")
        
        logger.info("\n⚖️  Métricas de Riesgo:")
        for metric in risk_metrics:
            if metric in metrics:
                logger.info(f"   {metric.replace('_', ' ').title()}: {metrics[metric]:.6f}")

        if "sharpe_gross" in metrics and "sharpe_net" in metrics:
            logger.info("\n💰 Costos BMV (bruto vs neto):")
            logger.info(f"   Sharpe Bruto: {metrics['sharpe_gross']:.6f}")
            logger.info(f"   Sharpe Neto:  {metrics['sharpe_net']:.6f}")
            if "win_rate_net" in metrics:
                logger.info(f"   Win Rate Neto: {metrics['win_rate_net']*100:.2f}%")
            if "round_trip_cost" in metrics:
                logger.info(f"   Round-trip: {metrics['round_trip_cost']*100:.3f}%")
        
        logger.info("=" * 50)
