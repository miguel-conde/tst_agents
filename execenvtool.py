import sys
import json

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
            exec(code, {}, self.environment)
            return self.environment
        except Exception as e:
            return {"error": str(e)}

    def get_environment(self):
        """Devuelve el entorno actual de variables."""
        return self.environment

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
            return {"error": f"Variable '{var_name}' no encontrada en el entorno."}

        try:
            variable = self.environment[var_name]
            size_in_bytes = sys.getsizeof(variable)

            # Umbral para truncar objetos grandes (por ejemplo, 1 MB)
            max_size = 1 * 1024 * 1024

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

    # Obtener el entorno actual
    print("Entorno final:", env.get_environment())
