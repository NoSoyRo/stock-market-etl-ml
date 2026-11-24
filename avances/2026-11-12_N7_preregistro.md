# Pre-registro · N7 · máscara de calidad desactualizada

Escrito el 2026-11-12 **antes de regenerar la máscara y antes de re-correr nada**. Rama `build/n-series`.
Desviaciones se anotan en el reporte, no se edita este texto.

## Qué es
**Prueba de sensibilidad, no idea nueva.** Mismas 5 reglas (E-frozen, E-refit, R-frozen, R-refit, REV), mismos parámetros,
misma ventana (señales 2026-06-19 → 2026-10-21), mismo criterio del preregistro `2026-11-09_oos_v2_preregistro.md`.
Solo cambia la máscara: se regenera con el store actual (cubre los 4 tickers nuevos y fechas posteriores a 2026-06-26).
No se cambian reglas ni umbrales de `quality.py`.

## Criterio: "la conclusión cambia" si y solo si
(a) algún veredicto de las 5 reglas pasa de "falla grosera" a otra categoría, o
(b) el IC de REV cambia de signo, o
(c) algún IC (de las 5) cambia en más de 0.03 en valor absoluto.
Si ninguna se cumple: "no cambia".

## Comprobaciones previas
- Causalidad: para tickers ya presentes, banderas con fecha ≤ 2026-06-26 de la máscara nueva = las de la vieja. Si difieren, se detiene todo y se reporta.
- Se documenta la fracción de filas excluida por la máscara nueva en la ventana OOS y en el train (< 2023-03-09), por ticker.
- Se mide cuántos cierres repetidos de AMXB (2015-18) quedan sin marcar (solo reporte; las reglas no se tocan).

## Ensayos
5 filas `log_trial`, familia `oos_v2_remask`. No suman al N de otras familias.
