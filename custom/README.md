# Custom Configs and Loaders

This directory lets you extend TopoExplorer with your own datasets, liftings, and
loaders — no code changes required. Files placed here are merged automatically with
the built-in TopoBench configs and loaders when the app starts.

---

## Directory structure

```
custom/
  configs/
    dataset/
      <domain>/
        <dataset_name>.yaml     ← one YAML per custom dataset
    transforms/
      liftings/
        <source>2<target>/
          <lifting_name>.yaml   ← one YAML per custom lifting
  loaders/
    my_loader.py                ← one or more Python files with loader classes
```

---

## Adding a custom dataset

### 1. Create a loader class (if needed)

If your dataset is not already supported by TopoBench, add a Python file to
`custom/loaders/`. The class name must end in `DatasetLoader` and must implement:

```python
from omegaconf import DictConfig

class MyCustomDatasetLoader:
    """Loads my custom dataset."""

    def __init__(self, parameters: DictConfig):
        self.parameters = parameters

    def load(self):
        """Return (data, dataset_dir).

        Returns
        -------
        data : torch_geometric.data.Data or similar
            The loaded dataset object.
        dataset_dir : pathlib.Path or str
            Directory where the raw/processed data lives.
        """
        # ... your loading logic ...
        return data, dataset_dir
```

The loader is discovered automatically — no registration step needed.

### 2. Create the dataset YAML

Add a YAML file to `custom/configs/dataset/<domain>/<dataset_name>.yaml`.
The structure mirrors TopoBench dataset configs:

```yaml
# custom/configs/dataset/graph/my_dataset.yaml

loader:
  _target_: MyCustomDatasetLoader   # class name from custom/loaders/
  parameters:
    data_dir: ${paths.data_dir}      # resolved to <repo_root>/datasets/
    # ... other loader parameters ...

parameters:
  num_features: 128
  num_classes: 7
  task_variable: y
  task_level: graph
  # ...

split_params:
  learning_setting: inductive
  # ...
```

The `${paths.data_dir}` interpolation resolves to `<topoexplorer_repo>/datasets/`.

Your dataset will appear in the UI under its `<domain>` with a **(custom)** label.

---

## Adding a custom lifting

Add a YAML file to `custom/configs/transforms/liftings/<source>2<target>/`.
The folder name (e.g. `graph2simplicial`) determines how the lifting is categorised
in the UI. The YAML structure mirrors TopoBench lifting configs:

```yaml
# custom/configs/transforms/liftings/graph2simplicial/my_lifting.yaml

transform_type: lifting
transform_name: MyLifting

# ... lifting-specific parameters ...
```

Your lifting will appear in the liftings panel with a **(custom)** label.

---

## Notes

- Custom configs are checked **before** TopoBench configs when resolving a dataset
  by name, so you can override a built-in dataset config by using the same domain
  and dataset name.
- Custom loaders are tried as a **fallback** after TopoBench built-in loaders.
- The `custom/` directory is not shipped with TopoExplorer — add it to `.gitignore`
  if you want to keep your configs private, or track it if you want to share them
  with your team.
