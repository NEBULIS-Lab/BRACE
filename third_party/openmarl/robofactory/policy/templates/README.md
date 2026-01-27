# Adding New VLA Policies to OpenMARL

This guide explains how to add a new VLA (Vision-Language-Action) policy to the OpenMARL framework.

## Overview

The OpenMARL policy framework provides base classes and utilities that make it easy to add new VLA policies while ensuring consistent interfaces, reducing code duplication, and enabling unified evaluation.

## Directory Structure

When adding a new policy, create the following structure:

```
robofactory/policy/
├── YourPolicy/
│   ├── yourpolicy_policy/
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   └── robot_yourpolicy.yaml
│   │   ├── dataset/
│   │   │   └── your_dataset.py
│   │   ├── model/
│   │   │   └── your_model_wrapper.py
│   │   ├── policy/
│   │   │   └── yourpolicy.py
│   │   └── workspace/
│   │       └── yourpolicy_workspace.py
│   ├── train.py
│   └── eval_multi_yourpolicy.py
```

## Step-by-Step Guide

### Step 1: Create the Policy Class

Inherit from `BaseVLAPolicy` in `robofactory.policy.core`:

```python
from robofactory.policy.core import BaseVLAPolicy

class YourPolicy(BaseVLAPolicy):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        action_dim: int = 8,
        **kwargs,
    ):
        super().__init__(device=device, action_dim=action_dim)
        # Initialize your model here
        self.model = YourModel(...)
    
    def predict_action(self, observation, instruction=None):
        # Implement action prediction
        return action
    
    def reset(self):
        # Reset any internal state
        pass
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cuda:0", **kwargs):
        # Factory method to load from checkpoint
        return cls(checkpoint_path=checkpoint_path, device=device, **kwargs)
```

### Step 2: Create the Workspace Class

Inherit from `BaseVLAWorkspace` in `robofactory.policy.core`:

```python
from robofactory.policy.core import BaseVLAWorkspace

class YourPolicyWorkspace(BaseVLAWorkspace):
    def _init_model(self):
        # Initialize and return your model
        model = YourModel(...)
        return model.to(self.device)
    
    def _init_dataset(self):
        # Create train and validation datasets
        train_dataset = YourDataset(train=True, ...)
        val_dataset = YourDataset(train=False, ...)
        return train_dataset, val_dataset
    
    def _compute_loss(self, batch):
        # Compute and return the training loss
        outputs = self.model(batch['image'], batch['action'])
        loss = outputs['loss']
        return loss
```

### Step 3: Create the Dataset Class

Inherit from `BaseVLADataset` in `robofactory.policy.core`:

```python
from robofactory.policy.core import BaseVLADataset

class YourDataset(BaseVLADataset):
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return dict with 'image', 'state', 'action', 'instruction'
        return {
            'image': self.data[idx]['image'],
            'state': self.data[idx]['state'],
            'action': self.data[idx]['action'],
            'instruction': self.data[idx]['instruction'],
        }
    
    def get_statistics(self):
        # Return normalization statistics
        return {
            'action': {'mean': ..., 'std': ..., 'q01': ..., 'q99': ...},
            'state': {'mean': ..., 'std': ..., 'q01': ..., 'q99': ...},
        }
```

### Step 4: Create Configuration File

Create a Hydra config inheriting from base configs:

```yaml
# yourpolicy_policy/config/robot_yourpolicy.yaml
defaults:
  - /core/config/base_training
  - /core/config/base_logging
  - _self_

# Policy-specific configuration
_target_: yourpolicy_policy.workspace.yourpolicy_workspace.YourPolicyWorkspace

exp_name: yourpolicy
agent_id: ${agent_id}

task:
  name: ${task_name}
  data_dir: data/lerobot_data/${task.name}_Agent${agent_id}_${data_num}

model:
  # Your model-specific parameters
  hidden_dim: 256
  num_layers: 4
  
training:
  learning_rate: 1.0e-4
  num_epochs: 100
```

### Step 5: Create Training Script

```python
# train.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="yourpolicy_policy/config", config_name="robot_yourpolicy")
def main(cfg: DictConfig):
    workspace = hydra.utils.instantiate(cfg, _recursive_=False)
    workspace.run()

if __name__ == "__main__":
    main()
```

### Step 6: Create Evaluation Script

Use the shared evaluation framework:

```python
from robofactory.policy.shared.evaluation import BasePolicyWrapper, BaseEvaluator

class YourPolicyWrapper(BasePolicyWrapper):
    def load_policy(self, checkpoint_path, **kwargs):
        self.policy = YourPolicy.from_checkpoint(checkpoint_path, ...)
    
    def get_action(self, observation=None):
        action = self.policy.predict_action(observation)
        return [action for _ in range(6)]  # Repeat for multi-step execution
```

## Using Shared Utilities

### Task Instructions

```python
from robofactory.policy.shared import get_task_instruction

instruction = get_task_instruction("LiftBarrier-rf", policy_type="detailed")
```

### Action Normalization

```python
from robofactory.policy.shared import normalize_action, denormalize_action

# Gaussian normalization
norm_action = normalize_action(action, mean, std)
action = denormalize_action(norm_action, mean, std)

# Quantile normalization
from robofactory.policy.shared import normalize_quantile, denormalize_quantile
norm_data = normalize_quantile(data, q01, q99)
```

### Image Processing

```python
from robofactory.policy.shared import process_image, hwc_to_chw

processed = process_image(image, target_size=(224, 224), augment=False)
```

## Best Practices

1. **Use Base Classes**: Always inherit from the core base classes to ensure interface consistency.

2. **Use Shared Utilities**: Use the shared utilities in `robofactory.policy.shared` instead of duplicating code.

3. **Checkpoint Compatibility**: Implement `from_checkpoint` class method for easy loading.

4. **Configuration**: Use Hydra defaults to inherit from base configs.

5. **Documentation**: Document your policy's specific requirements and capabilities.

6. **Testing**: Test with the unified evaluation framework before deployment.

## Example: Minimal New Policy

See `new_policy/` directory for a minimal working example that you can copy and modify.

