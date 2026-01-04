import os
import pathlib
from typing import Callable, Optional, Tuple, List, Dict
from torch.utils.data import Dataset

class AshFolderDataset(Dataset):
    """
    A generic dataset that loads files from a directory structure:
    root/
      class_a/
        file1.ext
        file2.ext
      class_b/
        file3.ext
    """
    def __init__(self, 
                 root: str, 
                 loader: Callable[[str], any], 
                 extensions: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Args:
            root (str): Root directory path.
            loader (Callable): A function that takes a path and returns data (e.g. PIL image).
            extensions (tuple, optional): Allowed file extensions (e.g. ('.jpg', '.png')).
            transform (Callable, optional): Transform to apply to the data.
            target_transform (Callable, optional): Transform to apply to the label.
        """
        self.root = pathlib.Path(root)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = extensions

        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = self._make_dataset(self.root, self.class_to_idx, self.extensions)

    def _find_classes(self, dir: pathlib.Path) -> Tuple[List[str], Dict[str, int]]:
        """Finds class names by scanning subdirectories."""
        classes = sorted([entry.name for entry in os.scandir(dir) if entry.is_dir()])
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folders in {dir}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir: pathlib.Path, class_to_idx: Dict[str, int], extensions: tuple) -> List[Tuple[str, int]]:
        """Creates a list of (path, class_index) tuples."""
        instances = []
        dir = dir.expanduser()
        
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = dir / target_class
            
            if not target_dir.is_dir():
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if extensions is None or fname.lower().endswith(extensions):
                        path = os.path.join(root, fname)
                        instances.append((path, class_index))
        
        return instances

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[any, int]:
        """
        Returns:
            (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)    
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target