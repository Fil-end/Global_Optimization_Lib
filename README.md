# Global_Optimization_Lib

  &emsp;The `global_optimization` package is a Python library designed for restoration of global optimization algorithms.  
  &emsp;Currently, the io between `ase.Atoms` and `mealpy` project (a library which restores comprehensive global optimization algorithms) has been preliminarily added into this project for further research.

## Currently restored

​ `ga`: `src/ga/ga.py`  
  `ssw`: `src/ssw/main.py`  
  `asop`: `src/asop/asop.py`  
  `gwo`: `src/gwo/GWO.py`  
  `mealpy`: `src/mealpy_io`
  
## Tips

  &emsp;Here we bind `calculator_method:LASP` to `SSW` while `calculator_method: MACE` to `GA`. (Currently, our main choice in `asop` is `calculator_method: LASP` to `SSW`, because `SSW` is a much more efficient global optimization algorithm than `GA`).

## Developing  

  &emsp;Here, the I/O between `ase.Atoms` and `mealpy` has been already written and debugged, my further idea on this may reopen a novel repo for `mealpy` and that may be called `MARGO` (Mealpy-based Atomic stRuctural Global Optimization). My further research will be focused on the comparison between selected global optimization algorithms in `mealpy` and other currently popular methods (including USPEX, Calypso and SSW maybe?). After all these comparison on efficiency and performance, I will valid the perforcement on realistic system (including `Surfaces`: metal-oxides (metal-inorganic binary systems), metal-metal binary systems, and further `Cluster`: nanoparticles system).

## Importance

  Here the `asop` method is just a copyright version by [Filend](https://github.com/Fil-end), you may find the official edition from [ZhiPan Liu Group](https://zpliu.fudan.edu.cn/).
