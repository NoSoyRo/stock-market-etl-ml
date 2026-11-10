#!/usr/bin/env python3
"""
Dashboard Web para monitorear Algoritmo Genético en tiempo real
Accesible desde cualquier dispositivo en la red local
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify
from dotenv import load_dotenv

# Directorio base del dashboard
DASHBOARD_DIR = Path(__file__).parent.absolute()

# Agregar src al path
sys.path.insert(0, str(DASHBOARD_DIR.parent.parent))

# Cargar .env desde la raíz del proyecto
PROJECT_ROOT = DASHBOARD_DIR.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

# Crear app con path explícito a templates
app = Flask(__name__, template_folder=str(DASHBOARD_DIR / 'templates'))

# News workspace also runs standalone through scripts/run_news_dashboard.py.
from src.dashboard.news import news, configure_news
configure_news(app)
app.register_blueprint(news, url_prefix='/news')


def get_db_connection():
    """Crear conexión a SQL Server"""
    # News is local and does not need the system ODBC driver to serve its UI.
    import pyodbc
    server = os.getenv('SQL_SERVER', 'localhost')
    database = os.getenv('SQL_DATABASE', 'MercadoBursatil')
    username = os.getenv('SQL_USER', '')
    password = os.getenv('SQL_PASSWORD', '')
    
    if username and password:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password}"
        )
    else:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
        )
    
    return pyodbc.connect(conn_str)


@app.route('/')
def index():
    """Página principal del dashboard"""
    return render_template('index.html')


@app.route('/api/experiments')
def get_experiments():
    """Obtener lista de experimentos"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            e.experiment_id,
            e.experiment_name,
            e.status,
            e.start_time,
            e.end_time,
            t.emisora_serie as ticker,
            e.total_models_trained as models_trained,
            (SELECT MAX(generation_number) FROM ml.experiment_generations WHERE experiment_id = e.experiment_id) as current_generation,
            e.population_size,
            e.num_generations
        FROM ml.experiments e
        LEFT JOIN stage.cat_tickers t ON e.ticker_id = t.ticker_id
        ORDER BY e.experiment_id DESC
        """
        
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        experiments = []
        
        for row in cursor.fetchall():
            exp = dict(zip(columns, row))
            # Convertir datetime a string
            if exp.get('start_time'):
                exp['start_time'] = exp['start_time'].strftime('%Y-%m-%d %H:%M:%S')
            if exp.get('end_time'):
                exp['end_time'] = exp['end_time'].strftime('%Y-%m-%d %H:%M:%S')
            experiments.append(exp)
        
        cursor.close()
        conn.close()
        
        return jsonify(experiments)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiment/<int:experiment_id>')
def get_experiment_detail(experiment_id):
    """Obtener detalle de un experimento específico"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Info del experimento
        cursor.execute("""
            SELECT 
                e.*,
                t.emisora_serie as ticker
            FROM ml.experiments e
            LEFT JOIN stage.cat_tickers t ON e.ticker_id = t.ticker_id
            WHERE e.experiment_id = ?
        """, (experiment_id,))
        
        row = cursor.fetchone()
        if not row:
            return jsonify({'error': 'Experiment not found'}), 404
        
        columns = [column[0] for column in cursor.description]
        experiment = dict(zip(columns, row))
        
        # Generaciones
        cursor.execute("""
            SELECT 
                generation_number,
                best_fitness,
                avg_fitness,
                worst_fitness,
                created_at
            FROM ml.experiment_generations
            WHERE experiment_id = ?
            ORDER BY generation_number
        """, (experiment_id,))
        
        generations = []
        for row in cursor.fetchall():
            gen = {
                'generation': row[0],
                'best_fitness': float(row[1]) if row[1] else None,
                'avg_fitness': float(row[2]) if row[2] else None,
                'worst_fitness': float(row[3]) if row[3] else None,
                'created_at': row[4].strftime('%H:%M:%S') if row[4] else None
            }
            generations.append(gen)
        
        # Top 10 modelos
        cursor.execute("""
            SELECT TOP 10
                model_id,
                val_rmse,
                val_mae,
                val_r2,
                val_sharpe_ratio,
                val_win_rate,
                JSON_VALUE(hyperparameters, '$.sequence_length') as seq_len,
                JSON_VALUE(hyperparameters, '$.hidden_units') as hidden,
                JSON_VALUE(hyperparameters, '$.num_layers') as layers,
                created_at
            FROM ml.model_registry
            WHERE experiment_id = ?
            ORDER BY val_rmse ASC
        """, (experiment_id,))
        
        models = []
        for row in cursor.fetchall():
            model = {
                'model_id': row[0],
                'val_rmse': float(row[1]) if row[1] else None,
                'val_mae': float(row[2]) if row[2] else None,
                'val_r2': float(row[3]) if row[3] else None,
                'sharpe': float(row[4]) if row[4] else None,
                'win_rate': float(row[5]) * 100 if row[5] else None,
                'seq_len': row[6],
                'hidden': row[7],
                'layers': row[8],
                'created_at': row[9].strftime('%H:%M:%S') if row[9] else None
            }
            models.append(model)
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'experiment': experiment,
            'generations': generations,
            'top_models': models
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/live')
def get_live_stats():
    """Obtener estadísticas en vivo del experimento activo"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Experimento más reciente activo
        cursor.execute("""
            SELECT TOP 1
                e.experiment_id,
                e.experiment_name,
                e.status,
                e.start_time,
                t.emisora_serie as ticker,
                e.population_size,
                e.num_generations
            FROM ml.experiments e
            LEFT JOIN stage.cat_tickers t ON e.ticker_id = t.ticker_id
            ORDER BY e.experiment_id DESC
        """)
        
        row = cursor.fetchone()
        if not row:
            return jsonify({'status': 'no_experiments'})
        
        experiment = {
            'id': row[0],
            'description': row[1],
            'status': row[2],
            'started': row[3].strftime('%Y-%m-%d %H:%M') if row[3] else None,
            'ticker': row[4],
            'pop_size': int(row[5]) if row[5] else 12,
            'num_gen': int(row[6]) if row[6] else 8
        }
        
        # Última generación
        cursor.execute("""
            SELECT TOP 1
                generation_number,
                best_fitness,
                avg_fitness
            FROM ml.experiment_generations
            WHERE experiment_id = ?
            ORDER BY generation_number DESC
        """, (experiment['id'],))
        
        gen_row = cursor.fetchone()
        if gen_row:
            experiment['current_gen'] = gen_row[0]
            experiment['best_fitness'] = float(gen_row[1]) if gen_row[1] else None
            experiment['avg_fitness'] = float(gen_row[2]) if gen_row[2] else None
        else:
            experiment['current_gen'] = 0
            experiment['best_fitness'] = None
            experiment['avg_fitness'] = None
        
        # Total de modelos entrenados
        cursor.execute("""
            SELECT COUNT(*) FROM ml.model_registry WHERE experiment_id = ?
        """, (experiment['id'],))
        experiment['models_trained'] = cursor.fetchone()[0]
        
        # Mejor modelo
        cursor.execute("""
            SELECT TOP 1
                model_id,
                val_rmse,
                val_sharpe_ratio,
                val_win_rate
            FROM ml.model_registry
            WHERE experiment_id = ?
            ORDER BY val_rmse ASC
        """, (experiment['id'],))
        
        best_row = cursor.fetchone()
        if best_row:
            experiment['best_model'] = {
                'id': best_row[0],
                'rmse': float(best_row[1]) if best_row[1] else None,
                'sharpe': float(best_row[2]) if best_row[2] else None,
                'win_rate': float(best_row[3]) * 100 if best_row[3] else None
            }
        
        # Progreso
        total_models = experiment['pop_size'] * experiment['num_gen']
        experiment['progress_pct'] = min(100, (experiment['models_trained'] / total_models) * 100)
        
        cursor.close()
        conn.close()
        
        return jsonify(experiment)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Correr en 0.0.0.0 para que sea accesible desde otros dispositivos
    print("\n" + "=" * 60)
    print("🚀 GA Dashboard iniciado")
    print("=" * 60)
    print(f"\n📱 Accede desde tu iPhone:")
    print(f"   http://<IP_DEL_SERVIDOR>:5000")
    print(f"\n💻 O desde este equipo:")
    print(f"   http://localhost:5000")
    print("\n" + "=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
