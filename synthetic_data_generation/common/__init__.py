"""Common modules for TEF integration — synthetic_data_generation service."""

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
    execute_GenerateData,
    execute_ListModels,
    execute_GetModelInfo,
    synthetic_data_generation_handlers,
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
    "execute_GenerateData",
    "execute_ListModels",
    "execute_GetModelInfo",
    "synthetic_data_generation_handlers",
]
