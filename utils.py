import os

def get_env_variable(var_name: str, cast_to=str, required=True):
    """
    Retrieve an environment variable and optionally cast its value to a specified type.

    Args:
        var_name (str): The name of the environment variable to retrieve.
        cast_to (type, optional): The type to cast the environment variable's value to. Defaults to str.
        required (bool, optional): Whether the environment variable is required. If True and the variable is not set, a RuntimeError is raised. Defaults to True.

    Returns:
        Any: The value of the environment variable, cast to the specified type if provided.

    Raises:
        RuntimeError: If the environment variable is required and not set.
    """
    value = os.getenv(var_name)
    if required and value is None:
        raise RuntimeError(f"Error: La variable de entorno {var_name} no está configurada.")
    if cast_to and value is not None:
        return cast_to(value)
    return value