WEBSOCKET_TYPE_RECEIVE = "CHAT/RECEIVE"
WEBSOCKET_TYPE_SEND = "CHAT/SEND"
WEBSOCKET_TYPE_SESSION = "CHAT/SESSION"
WEBSOCKET_TYPE_ERROR = "CHAT/ERROR"
WEBSOCKET_TYPE_COT = "CHAT/COT"
WEBSOCKET_TYPE_COLLECTION = "CHAT/COLLECTION"
WEBSOCKET_TYPE_STREAM = "CHAT/STREAM"
WEBSOCKET_TYPE_STREAM_AUDIO = "CHAT/STREAM_AUDIO"
WEBSOCKET_TYPE_STREAM_READY = "CHAT/STREAM_READY"
WEBSOCKET_TYPE_TURN_STATUS = "CHAT/TURN_STATUS"
WEBSOCKET_TYPE_STREAM_INFO = "CHAT/STREAM_INFO"
WEBSOCKET_TYPE_OPERATION_DATA = "CHAT/OPERATION_DATA"
WEBSOCKET_TYPE_MAP_CONTEXT = "CHAT/MAP_CONTEXT"
WEBSOCKET_TYPE_PATCH = "CHAT/PATCH"
WEBSOCKET_TYPE_THINKING = "CHAT/THINKING"
WEBSOCKET_TYPE_PROGRESS = "CHAT/PROGRESS"  # Streaming progress phases
WEBSOCKET_TYPE_PING = "CHAT/PING"
WEBSOCKET_TYPE_PONG = "CHAT/PONG"
WEBSOCKET_TYPE_TASK_PROGRESS = "CHAT/TASK_PROGRESS"  # Detailed task progress updates


# Progress Phase Constants
class ProgressPhase:
    """Progress phases for streaming updates to frontend."""
    UNDERSTANDING = "understanding"      # Analyzing the query
    ROUTING = "routing"                   # Determining which agent to use
    EXECUTING = "executing"              # Running tools/operations
    GENERATING = "generating"            # Generating response text


# Task Action Types (for Manus-style UI)
class TaskAction:
    """Task action types for detailed progress display."""
    THINKING = "thinking"
    SEARCHING = "searching"
    BROWSING = "browsing"
    CREATING = "creating"
    ANALYZING = "analyzing"
    GENERATING = "generating"


# Task Status
class TaskStatus:
    """Task status for progress tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"