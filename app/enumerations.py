from enum import IntEnum, Enum

class NumericLimits(IntEnum):
    """Centralizing numerical limits for workflows and invocations. """
    
    INVOCATION_LIMIT = 100
    RATE_RESET_TIME= 60
    BATCH_SIZE = 10
    SEMAPHORE_LIMIT = 15
    TIMEOUT = 10
    SHORT_SLEEP = 2
    BACKGROUND_INTERVAL = 600
    LONG_SLEEP = 60
    WARM_TIMESTAMP = 3600
    WARM_CHECK = 180
    BACKGROUND_INVOCATION_TRACK = 86400
    BACKGROUND_INDEX_TRACK = 3600
    TOOL_EXECUTION_POLL = 5
    TOOL_EXECUTION_INITIAL_WAIT = 1800
    TOOL_EXECUTION_EXTENSION = 300
    BATCH_LIMIT = 2

class TTLiveConfig(IntEnum):
    """ Centralizing TTL for invocation and worklows. """
    
    WORKFLOW_CACHE = 5
    INVOCATION_WORKFLOW_MAPPING = 120
    INVOCATION_LIST = 10
    RAW_INVOCATION_LIST = 10
    DUPLICATE_CHECK = 3
    INVOCACTION_RESULT = 259200  # 3 day cache for invocation result.
    SHORT_TTL = 20
    
class InvocationTracking(IntEnum):
    """ Invocation tracker time limits. """
    
    BASE_NO_PROGRESS = 12 * 3600      # 12h
    STALLED_THRESHOLD = 48 * 3600     # 3 days
    HARD_CAP = 7 * 24 * 3600          # 7 days

    POLL_QUICK = 10 # 10 sec
    POLL_FAST = 5 * 60 # 5 min
    POLL_MEDIUM = 10 * 60 # 10 min
    POLL_SLOW = 15 * 60 # 15 min
    POLL_MAX = 60 * 60 # 1 hour
    
class JobState(str, Enum):
    """ Galaxy job execution states. """
    
    NEW = "new"
    RESUBMITTED = "resubmitted"
    UPLOAD = "upload"
    WAITING = "waiting"
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    FAILED = "failed"
    PAUSED = "paused"
    DELETING = "deleting"
    DELETED = "deleted"
    STOPPING = "stop"
    STOPPED = "stopped"
    SKIPPED = "skipped"

class InvocationStates(Enum):
    PENDING = "Pending"
    COMPLETE = "Complete"
    FAILED = "Failed"

class SocketMessageType(str, Enum):
    """
    Enumeration of message types used for socket communication
    in workflow upload and execution, and tool execution processes.
    """

    # workflow upload
    
    TOOL_INSTALL = "TOOL_INSTALL"
    UPLOAD_WORKFLOW = "UPLOAD_WORKFLOW"
    UPLOAD_FAILURE = "UPLOAD_FAILURE"
    UPLOAD_COMPLETE = "UPLOAD_COMPLETE"
    
    # Workflow Invocation
    
    INVOCATION_FAILURE = "INVOCATION_FAILURE"
    INVOCATION_STEP_UPDATE = "INVOCATION_STEP_UPDATE"
    INVOCATION_COMPLETE = "INVOCATION_COMPLETE"

    # Workflow execution
    
    WORKFLOW_EXECUTE = "WORKFLOW_EXECUTE"
    WORKFLOW_FAILURE = "WORKFLOW_FAILURE"

    # Tool Execution
    
    JOB_UPDATE = "JOB_UPDATE"
    JOB_COMPLETE = "JOB_COMPLETE"
    JOB_FAILURE = "JOB_FAILURE"
    TOOL_EXECUTE = "TOOL_EXECUTE"
    
    # Output Dataset Indexing
    
    INDEX_UPDATE = "INDEX_UPDATE"
    INDEX_START =  "INDEX_START"
    INDEX_FINISH = "INDEX_FINISH"
    INDEX_FAIL = "INDEX_FAIL"

class SocketMessageEvent(str, Enum):
    """Enumeration of possible socket message event types for workflow and tool execution."""
    
    ping = "ping" # ping
    workflow_execute = "workflow_execute" # Workflow Execution   
    workflow_upload = "workflow_upload" # Workflow Execution   
    tool_execute = "tool_execute"  # Tool Execution    
    output_index = "output_index" # Output Indexing
    
class CollectionNames(Enum):
    """ Enumerations for mongo collection names. """
    
    DATA_ADOPTATION = "DataAdoptation"
    DATA_INDEXES = "DataIndexes"
    INVOCATION_LISTS = "InvocationList"
    INVOCATION_IDS = "InvocationIDs"
    INVOCATION_RESULTS = "InvocationResults"
    DELETED_INVOCATIONS = "DeletedInvocation"
    INVOCATION_STATES = "InvocationStates"
    INVOCATION_META = "InvocationMeta"
    GALAXY_THREAD_HISTORIES = "GalaxyThreadHistories"
    GALAXY_RUNS = "GalaxyRuns"
    GALAXY_RUN_REQUESTS = "GalaxyRunRequests"

class IndexingResponses(Enum):
    PENDING_MESSAGE = "Data Indexing in progress."
    COMPLETE_MESSAGE = "Indexing Completed."
    PENDING_STATUS = "Pending"
    COMPLETE_STATUS = "Complete"