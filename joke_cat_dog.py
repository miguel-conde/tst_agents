import pandas as pd

def que_paso_con_el_gato() -> str:
    """
    Responde a la pregunta sobre el gato.

    Returns:
        str: Una frase ocurrente respondiendo a la pregunta.
    """
    return "Se fue de parranda"

def que_paso_con_el_perro() -> str:
    """
    Responde a la pregunta sobre el perro.

    Returns:
        str: Otra frase ocurrente respondiendo a la pregunta.
    """
    return "El perro se comió la comida del gato"

def adstock(alpha: float, x: pd.Series):
    """
    Calcula el Adstock de una serie de datos.

    Args:
        alpha (float): Factor de atenuación.
        x (pd.Series): Serie de datos a procesar.

    Returns:
        pd.Series: Serie de Adstock.
    """
    return x.rolling(window=len(x), min_periods=1).apply(lambda y: (y * alpha ** pd.Series(range(len(y))[::-1])).sum())