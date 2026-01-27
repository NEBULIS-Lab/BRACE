"""
Base class for data format converters.

All data converters should inherit from this class and implement
the convert() method for their specific target format.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


class BaseDataConverter(ABC):
    """
    Abstract base class for data format converters.
    
    Subclasses should implement:
        - convert(): Main conversion logic
        - get_output_format(): Return the target format name
        
    Example:
        class ZarrToLeRobotConverter(BaseDataConverter):
            def convert(self, input_path, output_path, **kwargs):
                # Implementation
                pass
                
            def get_output_format(self):
                return 'lerobot'
    """
    
    @abstractmethod
    def convert(
        self,
        input_path: str,
        output_path: str,
        task_name: str,
        agent_id: int,
        **kwargs,
    ) -> str:
        """
        Convert data from source format to target format.
        
        Args:
            input_path: Path to input dataset (e.g., ZARR file)
            output_path: Path for output dataset
            task_name: Name of the task (e.g., 'LiftBarrier-rf')
            agent_id: Agent ID for the dataset
            **kwargs: Additional format-specific arguments
            
        Returns:
            Path to the created dataset
            
        Raises:
            ValueError: If input path doesn't exist or is invalid
            RuntimeError: If conversion fails
        """
        pass
    
    @abstractmethod
    def get_output_format(self) -> str:
        """
        Return the name of the output format.
        
        Returns:
            Format name string (e.g., 'lerobot', 'rlds', 'zarr-dp')
        """
        pass
    
    def validate_input(self, input_path: str) -> bool:
        """
        Validate that the input path exists and is a valid dataset.
        
        Args:
            input_path: Path to input dataset
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        path = Path(input_path)
        if not path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this converter.
        
        Returns:
            Dictionary with converter information
        """
        return {
            'class': self.__class__.__name__,
            'output_format': self.get_output_format(),
        }

