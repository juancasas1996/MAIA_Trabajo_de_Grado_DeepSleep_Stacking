"""El sistema como modelo de MLflow (pyfunc): el código de sistema/ + los archivos de modelo/ en un solo paquete."""
import mlflow.pyfunc


class SistemaSueno(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from sistema import cargar_sistema

        self.sistema = cargar_sistema(context.artifacts["modelo"])

    def predict(self, context, model_input, params=None):
        """model_input: dict o lista de dicts con "psg" (ruta local del EDF) y los campos opcionales de leer_noche."""
        from sistema import analizar

        entradas = model_input if isinstance(model_input, list) else [model_input]
        return [analizar(self.sistema, e) for e in entradas]
