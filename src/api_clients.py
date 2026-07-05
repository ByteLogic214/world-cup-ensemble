import requests
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class RateLimiter:
    """Implementa rate limiting para APIs"""
    
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Espera si se ha excedido el límite de llamadas"""
        now = time.time()
        # Limpiar llamadas antiguas (más de 1 minuto)
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit alcanzado. Esperando {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.calls = []
        
        self.calls.append(now)

class APIFootballClient:
    """Cliente mejorado para API-Football con retry y rate limiting"""
    
    def __init__(self, base_url: str, api_key: str, calls_per_minute: int = 30):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "x-apisports-key": api_key,
            "Accept": "application/json"
        }
        self.rate_limiter = RateLimiter(calls_per_minute)
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Crea una sesión con retry automático"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get(self, endpoint: str, params: dict) -> dict:
        """
        Realiza una petición GET con manejo de errores robusto
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            
        Returns:
            dict: Respuesta JSON de la API
            
        Raises:
            requests.RequestException: Si la petición falla
        """
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            logger.info(f"Llamando a API: {endpoint} con params: {params}")
            
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Validar estructura de respuesta
            if "response" not in data:
                logger.warning(f"Respuesta inesperada de API: {data}")
            
            return data
            
        except requests.Timeout:
            logger.error(f"Timeout al llamar a {endpoint}")
            raise
        except requests.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
            raise
        except requests.RequestException as e:
            logger.error(f"Error en petición a {endpoint}: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Error al parsear JSON: {str(e)}")
            raise
    
    def get_fixtures_by_league_season(self, league_id: int, season: int) -> List[dict]:
        """
        Obtiene fixtures de una liga y temporada específica
        
        Args:
            league_id: ID de la liga
            season: Año de la temporada
            
        Returns:
            List[dict]: Lista de fixtures
        """
        try:
            data = self._get("fixtures", {"league": league_id, "season": season})
            fixtures = data.get("response", [])
            
            logger.info(f"Obtenidos {len(fixtures)} fixtures para liga {league_id}, temporada {season}")
            
            return fixtures
            
        except Exception as e:
            logger.error(f"Error obteniendo fixtures: {str(e)}")
            return []
    
    def get_team_statistics(self, league_id: int, season: int, team_id: int) -> dict:
        """
        Obtiene estadísticas de un equipo
        
        Args:
            league_id: ID de la liga
            season: Año de la temporada
            team_id: ID del equipo
            
        Returns:
            dict: Estadísticas del equipo
        """
        try:
            data = self._get(
                "teams/statistics",
                {"league": league_id, "season": season, "team": team_id}
            )
            
            return data.get("response", {})
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas del equipo {team_id}: {str(e)}")
            return {}

class OddsAPIClient:
    """Cliente mejorado para The Odds API con rate limiting"""
    
    def __init__(self, base_url: str, api_key: str, calls_per_minute: int = 500):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limiter = RateLimiter(calls_per_minute)
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Crea una sesión con retry automático"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def get_h2h_odds(
        self,
        sport: str = "soccer_fifa_world_cup",
        regions: str = "eu",
        markets: str = "h2h"
    ) -> List[dict]:
        """
        Obtiene odds head-to-head
        
        Args:
            sport: Deporte a consultar
            regions: Regiones de bookmakers
            markets: Mercados a consultar
            
        Returns:
            List[dict]: Lista de eventos con odds
        """
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal"
        }
        
        try:
            logger.info(f"Obteniendo odds para {sport}")
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Verificar remaining requests en headers
            if "x-requests-remaining" in response.headers:
                remaining = response.headers["x-requests-remaining"]
                logger.info(f"Requests restantes: {remaining}")
            
            logger.info(f"Obtenidos odds para {len(data)} eventos")
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"Error obteniendo odds: {str(e)}")
            return []
