import logging
from src.main import train_pipeline
from src.settings import setup_logging

logger = logging.getLogger(__name__)

def main():
    """Punto de entrada principal de la aplicación"""
    try:
        # Configurar logging
        setup_logging(log_level="INFO")
        
        logger.info("="*80)
        logger.info("WORLD CUP ENSEMBLE - SISTEMA DE PREDICCIÓN")
        logger.info("="*80)
        
        # Ejecutar pipeline de entrenamiento
        predictor, metrics = train_pipeline()
        
        logger.info("\n" + "="*80)
        logger.info("APLICACIÓN FINALIZADA EXITOSAMENTE")
        logger.info("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nEjecución interrumpida por el usuario")
        return 130
        
    except Exception as e:
        logger.error(f"\nError fatal: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
