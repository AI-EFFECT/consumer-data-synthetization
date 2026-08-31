"""Common modules for TEF integration — synthetic_model_training service."""

from .task_manager import TaskManager, task_manager, get_task_manager
from .control_interface import (
    DataReference,
    ExecuteRequest,
    ExecuteResponse,
    StatusResponse,
    OutputResponse,
    create_control_router,
    create_app,
    run,
    get_data_url,
)
from .tef_operations import (
    execute_TrainModel,
    synthetic_model_training_handlers,
)

__all__ = [
    "TaskManager",
    "task_manager",
    "get_task_manager",
    "DataReference",
    "ExecuteRequest",
    "ExecuteResponse",
    "StatusResponse",
    "OutputResponse",
    "create_control_router",
    "create_app",
    "run",
    "get_data_url",
    "execute_TrainModel",
    "synthetic_model_training_handlers",
]
