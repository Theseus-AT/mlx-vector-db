# replication/replicator.py UND clustering/clustering_manager.py - Korrigierter Header
import asyncio
import json
import time
import os
import uuid
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict, field
import logging

# aioredis wird benötigt
try:
    import aioredis
except ImportError:
    print("Bitte 'aioredis' installieren: pip install aioredis")
    aioredis = None

logger = logging.getLogger("mlx_vector_db.clustering")

DEFAULT_NODE_ROLE = "replica"
DEFAULT_NODE_STATUS = "initializing"
NODE_HEARTBEAT_INTERVAL = 10  # Sekunden
NODE_TTL = NODE_HEARTBEAT_INTERVAL * 3 # Sekunden (Node gilt als inaktiv nach 3 Heartbeats)
LEADER_LOCK_KEY = "mlxvectordb:cluster:leader_lock"
LEADER_LOCK_TIMEOUT = NODE_HEARTBEAT_INTERVAL + 5 # Leader-Lock sollte etwas länger als Heartbeat sein

@dataclass
class NodeInfo:
    node_id: str
    host: str
    port: int
    role: str  # 'master' (oder 'leader'), 'replica' (oder 'follower')
    status: str  # 'active', 'inactive', 'initializing', 'unhealthy'
    last_heartbeat: float
    # Optional: zusätzliche Metadaten wie Load, verfügbare Stores etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClusterManager:
    def __init__(self, redis_url: str, node_id: Optional[str] = None):
        if aioredis is None:
            raise RuntimeError("aioredis ist nicht installiert. Clustering-Funktionen sind nicht verfügbar.")

        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None # Wird in start() initialisiert
        
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.current_node_info: Optional[NodeInfo] = None # Wird in _register_node() gesetzt
        
        self.cluster_nodes: Dict[str, NodeInfo] = {} # node_id -> NodeInfo
        self.is_leader = False
        
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._node_discovery_task: Optional[asyncio.Task] = None
        self._leader_election_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info(f"ClusterManager initialized for node_id: {self.node_id} (Redis: {self.redis_url})")

    async def _connect_redis(self):
        try:
            self.redis = await aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
            await self.redis.ping()
            logger.info("Successfully connected to Redis.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {self.redis_url}: {e}")
            self.redis = None # Sicherstellen, dass es None ist bei Fehler
            raise # Fehler weitergeben, damit Start fehlschlägt

    def _get_node_key(self, node_id: str) -> str:
        return f"mlxvectordb:cluster:node:{node_id}"

    async def _register_node(self):
        """Registriert diesen Node im Cluster oder aktualisiert seinen Status."""
        if not self.redis:
            logger.error("Cannot register node, Redis connection not available.")
            return

        host = os.getenv("NODE_HOST", "localhost") # Sollte die erreichbare IP/Hostname des Nodes sein
        port = int(os.getenv("NODE_PORT", "8000")) # Der Port, auf dem dieser Service läuft

        self.current_node_info = NodeInfo(
            node_id=self.node_id,
            host=host,
            port=port,
            role=DEFAULT_NODE_ROLE, # Startet standardmäßig als Replica
            status=DEFAULT_NODE_STATUS, # Initialisierungsstatus
            last_heartbeat=time.time(),
            metadata={"start_time": time.time()} # Beispiel Metadaten
        )
        
        try:
            node_data_json = json.dumps(asdict(self.current_node_info))
            await self.redis.setex(
                self._get_node_key(self.node_id),
                NODE_TTL,
                node_data_json
            )
            logger.info(f"Node {self.node_id} registered/heartbeat sent. Role: {self.current_node_info.role}, Status: {self.current_node_info.status}")
        except Exception as e:
            logger.error(f"Failed to register/heartbeat node {self.node_id}: {e}")


    async def _heartbeat_loop(self):
        """Sendet periodische Heartbeats an Redis."""
        while not self._shutdown_event.is_set():
            if self.current_node_info:
                self.current_node_info.last_heartbeat = time.time()
                # Status könnte hier basierend auf interner Logik aktualisiert werden
                # self.current_node_info.status = "active" # Wenn alles ok ist
                await self._register_node() # Sendet den aktuellen Zustand
            else:
                # Falls current_node_info noch nicht gesetzt wurde (z.B. bei initialer Registrierung)
                await self._register_node()

            await asyncio.sleep(NODE_HEARTBEAT_INTERVAL)
        logger.info("Heartbeat loop stopped.")

    async def _discover_nodes(self):
        """Entdeckt andere Nodes im Cluster durch Abfrage von Redis."""
        while not self._shutdown_event.is_set():
            if not self.redis:
                await asyncio.sleep(NODE_HEARTBEAT_INTERVAL)
                continue
            
            discovered_nodes: Dict[str, NodeInfo] = {}
            try:
                node_keys = await self.redis.keys(self._get_node_key("*"))
                for key in node_keys:
                    node_data_json = await self.redis.get(key)
                    if node_data_json:
                        try:
                            node_data = json.loads(node_data_json)
                            node_info = NodeInfo(**node_data)
                            # Ignoriere Nodes, deren TTL fast abgelaufen ist (könnten gerade verschwinden)
                            # Dies ist eine Vereinfachung; eine robustere Lösung würde TTL direkt von Redis prüfen.
                            if time.time() - node_info.last_heartbeat < NODE_TTL * 0.9:
                                discovered_nodes[node_info.node_id] = node_info
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse node data for key {key}: {e}")
                
                self.cluster_nodes = discovered_nodes
                if self.current_node_info and self.current_node_info.node_id in self.cluster_nodes:
                     self.current_node_info.status = "active" # Wenn er sich selbst sieht und alles gut ist

                # logger.debug(f"Discovered {len(self.cluster_nodes)} active nodes: {list(self.cluster_nodes.keys())}")
            except Exception as e:
                logger.error(f"Error during node discovery: {e}")
            
            await asyncio.sleep(NODE_HEARTBEAT_INTERVAL * 2) # Seltener als Heartbeat
        logger.info("Node discovery loop stopped.")

    async def _try_acquire_leader_lock(self) -> bool:
        """Versucht, den Leader-Lock in Redis zu setzen (atomar mit SET NX)."""
        if not self.redis:
            return False
        try:
            # SET key value NX EX timeout
            # NX: Nur setzen, wenn der Schlüssel noch nicht existiert.
            # EX: Setze Timeout in Sekunden.
            # Der Wert des Locks ist die ID des aktuellen Nodes.
            acquired = await self.redis.set(LEADER_LOCK_KEY, self.node_id, nx=True, ex=LEADER_LOCK_TIMEOUT)
            return bool(acquired)
        except Exception as e:
            logger.error(f"Error trying to acquire leader lock: {e}")
            return False

    async def _renew_leader_lock(self):
        """Erneuert den Leader-Lock, wenn dieser Node der Leader ist."""
        if not self.redis or not self.is_leader:
            return
        try:
            # Prüfen, ob der Lock noch diesem Node gehört, bevor er erneuert wird.
            current_leader = await self.redis.get(LEADER_LOCK_KEY)
            if current_leader == self.node_id:
                await self.redis.expire(LEADER_LOCK_KEY, LEADER_LOCK_TIMEOUT)
                # logger.debug("Leader lock renewed.")
            else:
                logger.warning(f"Lost leader lock or another node became leader. Current lock holder: {current_leader}")
                self.is_leader = False
                if self.current_node_info: self.current_node_info.role = DEFAULT_NODE_ROLE
        except Exception as e:
            logger.error(f"Error renewing leader lock: {e}")
            self.is_leader = False # Bei Fehler lieber aufgeben
            if self.current_node_info: self.current_node_info.role = DEFAULT_NODE_ROLE


    async def _leader_election_loop(self):
        """Führt periodisch Leader Election oder Lock-Erneuerung durch."""
        while not self._shutdown_event.is_set():
            if self.is_leader:
                await self._renew_leader_lock()
            else:
                # Versuche, Leader zu werden
                if await self._try_acquire_leader_lock():
                    self.is_leader = True
                    if self.current_node_info: self.current_node_info.role = "master" # Oder "leader"
                    logger.info(f"Node {self.node_id} became leader.")
                    # Hier könnten Aktionen für einen neuen Leader getriggert werden
                # else:
                    # logger.debug("Failed to acquire leader lock, another node is likely leader.")
            
            # Warte ein Intervall, das kürzer als die Lock-Timeout ist, um Renewals zu ermöglichen
            await asyncio.sleep(NODE_HEARTBEAT_INTERVAL)
        logger.info("Leader election loop stopped.")


    async def start(self):
        """Startet den ClusterManager und seine Hintergrundaufgaben."""
        if self.redis:
            logger.warning("ClusterManager seems to be already started or not properly shut down.")
            # Ggf. hier erst `stop()` aufrufen oder Fehler werfen.
            # Fürs Erste erlauben wir keinen Neustart ohne explizites stop().
            return

        try:
            await self._connect_redis()
            if not self.redis: # Wenn Verbindung fehlgeschlagen ist
                 raise ConnectionError("Failed to establish Redis connection during start.")
        except Exception as e:
            logger.critical(f"ClusterManager cannot start due to Redis connection failure: {e}")
            return # Nicht starten, wenn Redis nicht erreichbar ist

        await self._register_node() # Initiale Registrierung
        if self.current_node_info: self.current_node_info.status = "active" # Nach erfolgreicher Registrierung

        self._shutdown_event.clear() # Reset shutdown event

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._node_discovery_task = asyncio.create_task(self._discover_nodes())
        
        enable_leader_election = os.getenv("ENABLE_LEADER_ELECTION", "true").lower() == "true"
        if enable_leader_election:
            self._leader_election_task = asyncio.create_task(self._leader_election_loop())
            logger.info("Leader election enabled and started.")
        else:
            logger.info("Leader election is disabled.")


        logger.info(f"ClusterManager for node {self.node_id} started successfully.")

    async def stop(self):
        """Stoppt den ClusterManager und seine Hintergrundaufgaben sauber."""
        logger.info(f"Stopping ClusterManager for node {self.node_id}...")
        self._shutdown_event.set()

        tasks_to_wait_for = []
        if self._heartbeat_task: tasks_to_wait_for.append(self._heartbeat_task)
        if self._node_discovery_task: tasks_to_wait_for.append(self._node_discovery_task)
        if self._leader_election_task: tasks_to_wait_for.append(self._leader_election_task)

        if tasks_to_wait_for:
            await asyncio.gather(*tasks_to_wait_for, return_exceptions=True)
        
        if self.redis:
            # Optional: Node beim Herunterfahren deregistrieren oder als 'inactive' markieren
            if self.current_node_info:
                self.current_node_info.status = "inactive"
                try:
                    node_data_json = json.dumps(asdict(self.current_node_info))
                    # Hier mit kurzer TTL oder direkt löschen, je nach Strategie
                    await self.redis.setex(self._get_node_key(self.node_id), 10, node_data_json)
                except Exception as e:
                    logger.error(f"Error updating node status to inactive during shutdown: {e}")
            
            # Wenn dieser Node Leader war, den Lock freigeben
            if self.is_leader:
                try:
                    current_leader_in_redis = await self.redis.get(LEADER_LOCK_KEY)
                    if current_leader_in_redis == self.node_id:
                        await self.redis.delete(LEADER_LOCK_KEY)
                        logger.info("Released leader lock during shutdown.")
                except Exception as e:
                    logger.error(f"Error releasing leader lock during shutdown: {e}")

            await self.redis.close()
            self.redis = None # Verbindung schließen und zurücksetzen
            logger.info("Redis connection closed.")
        
        self.current_node_info = None
        self.is_leader = False
        logger.info(f"ClusterManager for node {self.node_id} stopped.")

    def get_all_nodes(self) -> List[NodeInfo]:
        """Gibt eine Liste aller bekannten aktiven Nodes im Cluster zurück."""
        return list(self.cluster_nodes.values())
    
    def get_leader_node_id(self) -> Optional[str]:
        """Gibt die ID des aktuellen Leader-Nodes zurück (basierend auf dem Lock)."""
        # Diese Funktion müsste den Lock in Redis prüfen, wenn der Manager selbst nicht der Leader ist.
        # Für eine einfache Implementierung geben wir nur zurück, ob dieser Node Leader ist.
        if self.is_leader:
            return self.node_id
        # Um den tatsächlichen Leader zu finden, müsste man LEADER_LOCK_KEY aus Redis lesen.
        # Dies ist hier nicht implementiert, um die Komplexität gering zu halten.
        # Die _leader_election_loop anderer Nodes würde versuchen, den Lock zu bekommen.
        logger.warning("get_leader_node_id currently only reflects if *this* node is leader. To find the actual leader, Redis must be queried for LEADER_LOCK_KEY.")
        return None # Oder hier Redis abfragen

    def get_replicas(self) -> List[NodeInfo]:
        """Gibt alle Nodes zurück, die als Replicas bekannt sind."""
        return [node for node in self.cluster_nodes.values() if node.role == "replica" and node.status == "active"]


# Beispiel für die Initialisierung und Verwendung (z.B. in main.py):
# async def main_application_startup(app: FastAPI):
#     config = get_config() # Ihre Konfigurationslogik
#     if config.clustering.enable_clustering: # Annahme: config.clustering Objekt
#         cluster_manager = ClusterManager(redis_url=config.clustering.redis_url)
#         app.state.cluster_manager = cluster_manager # Im App-State verfügbar machen
#         await cluster_manager.start()
#
# async def main_application_shutdown(app: FastAPI):
#     if hasattr(app.state, "cluster_manager") and app.state.cluster_manager:
#         await app.state.cluster_manager.stop()