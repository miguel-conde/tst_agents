import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from typing import List, Dict, Any

from tracer import tracer

class ExecutionEnvironment:
    def __init__(self):
        """Inicializa el entorno de ejecución con un diccionario vacío."""
        self.environment = {}

    def pyExec(self, code: str):
        """
        Ejecuta el código Python dado utilizando el entorno interno.
        
        Parameters:
        code (str): Código Python a ejecutar.

        Returns:
        dict: Diccionario con las variables actualizadas tras la ejecución.
        """
        try:
            # Ejecutar el código usando el entorno interno
            tracer.debug(f"Ejecutando código:\n\n{code}\n\n")
            # exec(code, {}, self.environment)
            exec(code, self.environment, self.environment)
            # return self.environment
            # return {var: self.format_variable(var) for var in self.environment}
            return f"Variables in environment: {[x for x in self.environment]}"
        except Exception as e:
            tracer.error(f"Error al ejecutar el código: {e}")
            return {"error": str(e)}

    def get_environment(self):
        """Devuelve el entorno actual de variables."""
        tracer.debug("Obteniendo el entorno actual.")

        out = dict()
        for var in self.environment:
            out[var] = self.format_variable(var)

        # return self.environment
        return out

    def format_variable_old(self, var_name: str):
        """
        Convierte una variable del entorno a un formato entendible por el LLM.
        Si la variable es demasiado grande, la convierte parcialmente.
        
        Parameters:
        var_name (str): Nombre de la variable a formatear.

        Returns:
        dict: Representación formateada de la variable o un error si no existe.
        """
        if var_name not in self.environment:
            tracer.error(f"Variable '{var_name}' no encontrada en el entorno.")
            return {"error": f"Variable '{var_name}' no encontrada en el entorno."}

        try:
            tracer.debug(f"Formateando variable '{var_name}'.")
            variable = self.environment[var_name]
            size_in_bytes = sys.getsizeof(variable)

            # Umbral para truncar objetos grandes (por ejemplo, 1 MB)
            max_size = 1 * 1024 * 1024

            if isinstance(variable, pd.DataFrame):
                # Convertir DataFrame a JSON-friendly si es demasiado grande
                if size_in_bytes > max_size:
                    truncated_df = variable.head(100).to_dict(orient="records")
                    return {
                        "type": "DataFrame",
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": truncated_df
                    }
                else:
                    return {
                        "type": "DataFrame",
                        "size": size_in_bytes,
                        "truncated": False,
                        "content": variable.to_dict(orient="records")
                    }

            if isinstance(variable, np.ndarray):
                # Convertir numpy array a JSON-friendly
                if size_in_bytes > max_size:
                    truncated_array = variable[:100].tolist()  # Truncar a los primeros 100 elementos
                    return {
                        "type": "ndarray",
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": truncated_array
                    }
                else:
                    return {
                        "type": "ndarray",
                        "size": size_in_bytes,
                        "truncated": False,
                        "content": variable.tolist()
                    }

            if isinstance(variable, (set, tuple)):
                # Convertir set o tuple a lista para JSON-friendly
                converted = list(variable)
                if size_in_bytes > max_size:
                    return {
                        "type": type(variable).__name__,
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": converted[:100]  # Truncar a los primeros 100 elementos
                    }
                return {
                    "type": type(variable).__name__,
                    "size": size_in_bytes,
                    "truncated": False,
                    "content": converted
                }

            if isinstance(variable, complex):
                # Convertir números complejos a representaciones JSON-friendly
                return {
                    "type": "complex",
                    "size": size_in_bytes,
                    "content": {
                        "real": variable.real,
                        "imag": variable.imag
                    }
                }

            if hasattr(variable, "evalf") and callable(getattr(variable, "evalf", None)):
                # Manejar objetos de SymPy convirtiéndolos a string
                return {
                    "type": "SymPy",
                    "size": size_in_bytes,
                    "content": str(variable)
                }

            if isinstance(variable, plt.Figure):
                # Manejar objetos de matplotlib (figuras)
                import io
                buffer = io.BytesIO()
                variable.savefig(buffer, format="png")
                buffer.seek(0)
                encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
                return {
                    "type": "matplotlib.Figure",
                    "size": len(encoded_image),
                    "content": encoded_image,
                    "format": "base64",
                    "description": "Imagen codificada en base64 para su representación."
                }

            if size_in_bytes > max_size:
                # Si es una lista o diccionario grande, truncarlo
                if isinstance(variable, (list, dict)):
                    truncated = variable[:100] if isinstance(variable, list) else dict(list(variable.items())[:100])
                    return {
                        "type": type(variable).__name__,
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": truncated
                    }
                else:
                    return {
                        "type": type(variable).__name__,
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": str(variable)[:1000]  # Truncar cadenas u objetos convertibles
                    }

            # Si es manejable, devolver completo
            return {
                "type": type(variable).__name__,
                "size": size_in_bytes,
                "content": variable
            }
        except Exception as e:
            return {"error": str(e)}

    def is_json_serializable(self, obj: Any) -> bool:
        """
        Verifica si un objeto es serializable a JSON.

        Args:
            obj (Any): Objeto a verificar.

        Returns:
            bool: True si el objeto es JSON-serializable, False en caso contrario.
        """
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    def format_variable(self, var_name: str):
        """
        Convierte una variable del entorno a un formato entendible por el LLM.
        Si la variable es demasiado grande, la convierte parcialmente.

        Parameters:
        var_name (str): Nombre de la variable a formatear.

        Returns:
        dict: Representación formateada de la variable o un error si no existe.
        """
        if var_name not in self.environment:
            tracer.error(f"Variable '{var_name}' no encontrada en el entorno.")
            return {"error": f"Variable '{var_name}' no encontrada en el entorno."}

        try:
            tracer.debug(f"Formateando variable '{var_name}'.")
            variable = self.environment[var_name]
            size_in_bytes = sys.getsizeof(variable)

            # Umbral para truncar objetos grandes (por ejemplo, 1 KB)
            max_size = 1 * 1024

            if isinstance(variable, pd.DataFrame):
                # Convertir DataFrame a JSON-friendly si es demasiado grande
                if size_in_bytes > max_size:
                    truncated_df = variable.head(100).to_dict(orient="records")
                    return {
                        "type": "DataFrame",
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": truncated_df
                    }
                else:
                    return {
                        "type": "DataFrame",
                        "size": size_in_bytes,
                        "truncated": False,
                        "content": variable.to_dict(orient="records")
                    }

            if isinstance(variable, np.ndarray):
                # Convertir numpy array a JSON-friendly
                if size_in_bytes > max_size:
                    truncated_array = variable[:100].tolist()  # Truncar a los primeros 100 elementos
                    return {
                        "type": "ndarray",
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": truncated_array
                    }
                else:
                    return {
                        "type": "ndarray",
                        "size": size_in_bytes,
                        "truncated": False,
                        "content": variable.tolist()
                    }

            if isinstance(variable, (set, tuple)):
                # Convertir set o tuple a lista para JSON-friendly
                converted = list(variable)
                if size_in_bytes > max_size:
                    return {
                        "type": type(variable).__name__,
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": converted[:100]  # Truncar a los primeros 100 elementos
                    }
                return {
                    "type": type(variable).__name__,
                    "size": size_in_bytes,
                    "truncated": False,
                    "content": converted
                }

            if isinstance(variable, complex):
                # Convertir números complejos a representaciones JSON-friendly
                return {
                    "type": "complex",
                    "size": size_in_bytes,
                    "content": {
                        "real": variable.real,
                        "imag": variable.imag
                    }
                }

            if hasattr(variable, "evalf") and callable(getattr(variable, "evalf", None)):
                # Manejar objetos de SymPy convirtiéndolos a string
                return {
                    "type": "SymPy",
                    "size": size_in_bytes,
                    "content": str(variable)
                }

            if isinstance(variable, plt.Figure):
                # Manejar objetos de matplotlib (figuras)
                import io
                buffer = io.BytesIO()
                variable.savefig(buffer, format="png")
                buffer.seek(0)
                encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
                return {
                    "type": "matplotlib.Figure",
                    "size": len(encoded_image),
                    "content": encoded_image,
                    "format": "base64",
                    "description": "Imagen codificada en base64 para su representación."
                }

            if size_in_bytes > max_size:
                # Si es una lista o diccionario grande, truncarlo
                if isinstance(variable, (list, dict)):
                    truncated = variable[:100] if isinstance(variable, list) else dict(list(variable.items())[:100])
                    return {
                        "type": type(variable).__name__,
                        "size": size_in_bytes,
                        "truncated": True,
                        "content": truncated
                    }
                else:
                    try:
                        return {
                            "type": type(variable).__name__,
                            "size": size_in_bytes,
                            "truncated": True,
                            "content": str(variable)[:1000]  # Truncar cadenas u objetos convertibles
                        }
                    except Exception:
                        return {
                            "type": type(variable).__name__,
                            "size": size_in_bytes,
                            "truncated": True,
                            "content": f"[Error al convertir {type(variable).__name__} a cadena]"
                        }

            # Validar si el objeto es JSON-serializable
            if self.is_json_serializable(variable):
                return {
                    "type": type(variable).__name__,
                    "size": size_in_bytes,
                    "content": variable
                }
            else:
                # Si no es serializable, devolver su representación en string
                try:
                    return {
                        "type": type(variable).__name__,
                        "size": size_in_bytes,
                        "content": str(variable)[:1000]  # Representación truncada en string
                    }
                except Exception:
                    return {
                        "type": type(variable).__name__,
                        "size": size_in_bytes,
                        "content": f"[Error al convertir {type(variable).__name__} a cadena]"
                    }

        except Exception as e:
            return {"error": str(e)}


# Ejemplo de uso
if __name__ == "__main__":


    env = ExecutionEnvironment()

    # Ejecutar una línea de código
    result1 = env.pyExec("x = [i for i in range(10000)]")
    print("Entorno después de la primera ejecución:", result1)

    # Formatear una variable grande
    formatted_var = env.format_variable("x")
    print("Variable 'x' formateada:", json.dumps(formatted_var, indent=2))

    # Usar las variables ya definidas
    result2 = env.pyExec("y = sum(x)")
    print("Entorno después de la segunda ejecución:", result2)

    # Formatear una variable pequeña
    formatted_var_y = env.format_variable("y")
    print("Variable 'y' formateada:", json.dumps(formatted_var_y, indent=2))

    # Manejar un DataFrame
    env.pyExec("import pandas as pd\ndata = pd.DataFrame({'col1': range(1000), 'col2': range(1000)})")
    formatted_df = env.format_variable("data")
    print("Variable 'data' formateada:", json.dumps(formatted_df, indent=2))

    # Manejar un numpy array
    env.pyExec("import numpy as np\nnarray = np.arange(10000)")
    formatted_array = env.format_variable("narray")
    print("Variable 'narray' formateada:", json.dumps(formatted_array, indent=2))

    # Manejar un set
    env.pyExec("my_set = {i for i in range(500)}")
    formatted_set = env.format_variable("my_set")
    print("Variable 'my_set' formateada:", json.dumps(formatted_set, indent=2))

    # Manejar un tuple
    env.pyExec("my_tuple = tuple(range(500))")
    formatted_tuple = env.format_variable("my_tuple")
    print("Variable 'my_tuple' formateada:", json.dumps(formatted_tuple, indent=2))

    # Manejar un número complejo
    env.pyExec("complex_num = 3 + 4j")
    formatted_complex = env.format_variable("complex_num")
    print("Variable 'complex_num' formateada:", json.dumps(formatted_complex, indent=2))

    # Manejar un objeto SymPy
    env.pyExec("from sympy import symbols\nz = symbols('z')\nexpression = z**2 + 3*z + 2")
    formatted_sympy = env.format_variable("expression")
    print("Variable 'expression' formateada:", json.dumps(formatted_sympy, indent=2))

    # Manejar una figura de matplotlib
    env.pyExec("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([0, 1], [0, 1])")
    formatted_fig = env.format_variable("fig")
    print("Variable 'fig' formateada:", json.dumps(formatted_fig, indent=2))

    # Manejar un objeto no serializable (archivo abierto)
    env.pyExec("file_obj = open('test_file.txt', 'w')")
    formatted_file = env.format_variable("file_obj")
    print("Variable 'file_obj' formateada:", json.dumps(formatted_file, indent=2))

    env.pyExec("import pandas as pd")
    formatted_pd = env.format_variable("pd")
    print("Variable 'pd' formateada:", json.dumps(formatted_pd, indent=2))

    # Obtener el entorno actual
    print("Entorno final:", env.get_environment())
