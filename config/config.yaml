import os
import yaml
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Configuración de logging profesional
def setup_logging(log_level: str = "INFO"):
    """Configura el sistema de logging de manera profesional"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler()
        ]
    )

@dataclass
class Secrets:
    """Clase para manejar secretos de manera segura"""
    api_football_key: str
    odds_api_key: str
    
    def validate(self) -> bool:
        """Valida que los secretos no estén vacíos"""
        if not self.api_football_key or not self.odds_api_key:
            return False
        return True

def load_config(path: str = "config/config.yaml") -> dict:
    """
    Carga la configuración desde un archivo YAML con validación
    
    Args:
        path: Ruta al archivo de configuración
        
    Returns:
        dict: Configuración cargada
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si el archivo YAML es inválido
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Validación básica de la estructura
        required_keys = ["api_football", "odds_api", "competition", "data", "model"]
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Faltan claves requeridas en config: {missing_keys}")
        
        return config
    
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error al parsear YAML: {e}")

def load_secrets() -> Secrets:
    """
    Carga los secretos desde variables de entorno con validación
    
    Returns:
        Secrets: Objeto con las credenciales
        
    Raises:
        ValueError: Si faltan credenciales críticas
    """
    secrets = Secrets(
        api_football_key=os.getenv("API_FOOTBALL_KEY", ""),
        odds_api_key=os.getenv("ODDS_API_KEY", "")
    )
    
    if not secrets.validate():
        raise ValueError(
            "Credenciales faltantes. Asegúrate de configurar:\n"
            "- API_FOOTBALL_KEY\n"
            "- ODDS_API_KEY"
        )
    
    return secrets

# Crear directorio de logs al importar
Path("logs").mkdir(exist_ok=True)
setup_logging()
