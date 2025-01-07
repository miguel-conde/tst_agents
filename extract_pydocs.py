import ast
from typing import List, Dict, Any

def extract_documentation(file_path: str) -> Dict[str, Any]:
    """
    Extrae la documentación de un archivo Python (.py) incluyendo docstrings de módulos, clases y funciones.

    Args:
        file_path (str): Ruta del archivo Python.

    Returns:
        Dict[str, Any]: Un diccionario con las docstrings organizadas por tipo (module, classes, functions).
    """
    documentation = {
        "module doc": "",
        "classes": {},
        "functions": {}
    }

    try:
        # Leer el contenido del archivo
        with open(file_path, "r", encoding="utf-8") as file:
            code = file.read()

        # Parsear el contenido a un árbol de sintaxis abstracta (AST)
        tree = ast.parse(code)

        # Extraer la docstring del módulo
        documentation["module"] = ast.get_docstring(tree)

        # Iterar por todos los nodos del AST
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # Extraer docstring de la clase
                class_doc = ast.get_docstring(node)
                methods = {}

                # Extraer docstrings de métodos dentro de la clase
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        methods[sub_node.name] = ast.get_docstring(sub_node)

                documentation["classes"][node.name] = {
                    "docstring": class_doc,
                    "methods": methods
                }

            elif isinstance(node, ast.FunctionDef):
                # Extraer docstring de funciones fuera de clases
                documentation["functions"][node.name] = ast.get_docstring(node)

    except Exception as e:
        raise RuntimeError(f"Error al procesar el archivo {file_path}: {e}")

    return documentation

# Ejemplo de uso
if __name__ == "__main__":
    import json

    # Ruta de ejemplo
    file_path = "llm_connector.py"  # Cambiar por la ruta del archivo que deseas analizar

    docs = extract_documentation(file_path)

    # Mostrar la documentación extraída de forma legible
    print(json.dumps(docs, indent=4, ensure_ascii=False))
