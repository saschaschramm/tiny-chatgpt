from typing import Optional
import uuid

class Node():

    def __init__(self, timesteps: int, t:int, identifier: int):
        self.timesteps: int = timesteps
        self.identifier: int = identifier
        self.t: int = t
        self.node_id: str = str(uuid.uuid4())
        self.children: Optional[list] = None
