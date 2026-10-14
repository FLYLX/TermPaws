import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from utils.logger import logger


class Subscriber:
    """Room 订阅者"""
    def __init__(self, sid: str, subscriber_type: str, user_uuid: str = "unknown", ip: str = "unknown"):
        self.sid = sid
        self.type = subscriber_type  # "backend" or "browser"
        self.user_uuid = user_uuid
        self.ip = ip
        self.connected_at = datetime.now()


class ItemRoom:
    """Item Room - 包含订阅者和输出缓存"""
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.permanent_subscribers: Dict[str, Subscriber] = {}  # sid -> Subscriber (Backend)
        self.temporary_subscribers: Dict[str, Subscriber] = {}  # sid -> Subscriber (Browser)
        self.output_cache: List[Dict[str, Any]] = []  # 断连时的输出缓存
        self.last_output_time: Optional[datetime] = None
        self.lock = threading.Lock()
    
    def add_subscriber(self, subscriber: Subscriber) -> bool:
        """添加订阅者"""
        with self.lock:
            if subscriber.type == "backend":
                # 一个 room 只认一条 backend 订阅：新 backend 进来时清掉旧的
                # （旧的多半是断连僵尸——backend 重启/重连后旧连接没被及时移除，
                # 留着会让每条输出被多个 backend 通道各转一份，日志重复 ×N）。
                zombies = [sid for sid in self.permanent_subscribers if sid != subscriber.sid]
                for sid in zombies:
                    self.permanent_subscribers.pop(sid)
                    logger.warning(
                        f"[Room {self.room_id}] Evicted stale backend subscriber {sid[:16]}... (replaced by {subscriber.sid[:16]}...)"
                    )
                self.permanent_subscribers[subscriber.sid] = subscriber
                logger.info(f"[Room {self.room_id}] Added permanent subscriber: {subscriber.sid[:16]}...")
            else:
                self.temporary_subscribers[subscriber.sid] = subscriber
                logger.info(f"[Room {self.room_id}] Added temporary subscriber: {subscriber.sid[:16]}... (user={subscriber.user_uuid})")
            return True
    
    def remove_subscriber(self, sid: str) -> Optional[Subscriber]:
        """移除订阅者"""
        with self.lock:
            if sid in self.permanent_subscribers:
                subscriber = self.permanent_subscribers.pop(sid)
                logger.info(f"[Room {self.room_id}] Removed permanent subscriber: {sid[:16]}...")
                return subscriber
            elif sid in self.temporary_subscribers:
                subscriber = self.temporary_subscribers.pop(sid)
                logger.info(f"[Room {self.room_id}] Removed temporary subscriber: {sid[:16]}... (user={subscriber.user_uuid})")
                return subscriber
            return None
    
    def get_all_subscribers(self) -> List[Subscriber]:
        """获取所有订阅者"""
        with self.lock:
            return list(self.permanent_subscribers.values()) + list(self.temporary_subscribers.values())
    
    def get_subscriber_sids(self) -> List[str]:
        """获取所有订阅者的 SID"""
        with self.lock:
            return list(self.permanent_subscribers.keys()) + list(self.temporary_subscribers.keys())
    
    def has_permanent_subscribers(self) -> bool:
        """是否有永久订阅者"""
        with self.lock:
            return len(self.permanent_subscribers) > 0
    
    def has_subscribers(self) -> bool:
        """是否有任何订阅者"""
        with self.lock:
            return len(self.permanent_subscribers) > 0 or len(self.temporary_subscribers) > 0
    
    def add_to_cache(self, output: Dict[str, Any]):
        """添加输出到缓存"""
        with self.lock:
            self.output_cache.append(output)
            self.last_output_time = datetime.now()
            if len(self.output_cache) > 1000:
                self.output_cache = self.output_cache[-500:]
    
    def get_cache(self) -> List[Dict[str, Any]]:
        """获取并清空缓存"""
        with self.lock:
            cache = self.output_cache.copy()
            self.output_cache = []
            return cache
    
    def get_subscriber_info(self) -> Dict[str, Any]:
        """获取订阅者信息"""
        with self.lock:
            return {
                "room_id": self.room_id,
                "permanent_count": len(self.permanent_subscribers),
                "temporary_count": len(self.temporary_subscribers),
                "permanent": [
                    {"sid": sid[:16] + "...", "type": s.type, "user_uuid": s.user_uuid, "ip": s.ip}
                    for sid, s in self.permanent_subscribers.items()
                ],
                "temporary": [
                    {"sid": sid[:16] + "...", "type": s.type, "user_uuid": s.user_uuid, "ip": s.ip}
                    for sid, s in self.temporary_subscribers.items()
                ]
            }


class DaemonRoomManager:
    """
    Daemon Room 管理器
    
    按 UPDATE.MD 规范实现 Room 机制：
    1. 每个 item_uuid 对应一个 Room
    2. Room 内区分 permanent（Backend）和 temporary（Browser）订阅者
    3. 支持输出缓存，断连重连时可恢复
    4. Item 停止时销毁 Room
    """
    def __init__(self):
        self.rooms: Dict[str, ItemRoom] = {}  # room_id -> ItemRoom
        self.lock = threading.Lock()
    
    def create_room(self, room_id: str) -> ItemRoom:
        """创建 Item Room"""
        with self.lock:
            if room_id not in self.rooms:
                self.rooms[room_id] = ItemRoom(room_id)
                logger.info(f"[RoomManager] Created room: {room_id}")
            return self.rooms[room_id]
    
    def get_room(self, room_id: str) -> Optional[ItemRoom]:
        """获取 Room"""
        with self.lock:
            return self.rooms.get(room_id)
    
    def room_exists(self, room_id: str) -> bool:
        """检查 Room 是否存在"""
        with self.lock:
            return room_id in self.rooms
    
    def add_subscriber(self, room_id: str, subscriber: Subscriber) -> bool:
        """添加订阅者到 Room"""
        with self.lock:
            room = self.rooms.get(room_id)
            if not room:
                logger.warning(f"[RoomManager] Room {room_id} not found, creating...")
                room = self.create_room(room_id)
            return room.add_subscriber(subscriber)
    
    def remove_subscriber(self, room_id: str, sid: str) -> Optional[Subscriber]:
        """从 Room 移除订阅者"""
        with self.lock:
            room = self.rooms.get(room_id)
            if not room:
                return None
            return room.remove_subscriber(sid)
    
    def remove_subscriber_from_any_room(self, sid: str) -> tuple[Optional[str], Optional[Subscriber]]:
        """从任意 Room 移除订阅者（用于断连处理）"""
        with self.lock:
            for room_id, room in self.rooms.items():
                subscriber = room.remove_subscriber(sid)
                if subscriber:
                    return (room_id, subscriber)
            return (None, None)
    
    def get_subscriber_sids(self, room_id: str) -> List[str]:
        """获取 Room 内所有订阅者的 SID"""
        with self.lock:
            room = self.rooms.get(room_id)
            if not room:
                return []
            return room.get_subscriber_sids()
    
    def add_output_to_cache(self, room_id: str, output: Dict[str, Any]):
        """添加输出到 Room 缓存"""
        with self.lock:
            room = self.rooms.get(room_id)
            if room:
                room.add_to_cache(output)
    
    def get_cached_output(self, room_id: str) -> List[Dict[str, Any]]:
        """获取 Room 缓存的输出"""
        with self.lock:
            room = self.rooms.get(room_id)
            if room:
                return room.get_cache()
            return []
    
    def destroy_room(self, room_id: str) -> List[str]:
        """
        销毁 Room（Item 停止时）
        
        Returns:
            被移除的订阅者 SID 列表
        """
        with self.lock:
            room = self.rooms.pop(room_id, None)
            if room:
                sids = room.get_subscriber_sids()
                logger.info(f"[RoomManager] Destroyed room {room_id}, removed {len(sids)} subscribers")
                return sids
            return []
    
    def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """获取 Room 信息"""
        with self.lock:
            room = self.rooms.get(room_id)
            if room:
                return room.get_subscriber_info()
            return None

    def has_permanent_subscribers(self, room_id: str) -> bool:
        """Return whether Backend is currently joined to the Item Room."""
        with self.lock:
            room = self.rooms.get(room_id)
        return bool(room and room.has_permanent_subscribers())
    
    def get_all_rooms_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 Room 信息"""
        with self.lock:
            return {room_id: room.get_subscriber_info() for room_id, room in self.rooms.items()}
    
    def get_permanent_subscriber_sids(self, room_id: str) -> List[str]:
        """获取 Room 内永久订阅者的 SID"""
        with self.lock:
            room = self.rooms.get(room_id)
            if room:
                return list(room.permanent_subscribers.keys())
            return []
    
    def get_temporary_subscriber_sids(self, room_id: str) -> List[str]:
        """获取 Room 内临时订阅者的 SID"""
        with self.lock:
            room = self.rooms.get(room_id)
            if room:
                return list(room.temporary_subscribers.keys())
            return []


room_manager = DaemonRoomManager()
