# Global_Optimization_Lib

The `global_optimization` package is a Python library designed for restoration of global optimization algorithms.

## Currently restored

​ `ga`: `src/ga/ga.py`  
  `ssw`: `src/ssw/main.py`  
  `asop`: `src/asop/asop.py`  
  `gwo`: `src/gwo/GWO.py`
  
## Tips

  Here we bind `calculator_method:LASP` to `SSW` while `calculator_method: MACE` to `GA`.  
    (Currently, our main choice in `asop` is `calculator_method: LASP` to `SSW`, because `SSW` is a much more efficient global optimization algorithm than `GA`).

## Developing  

  Here, the I/O between `ase.Atoms` and global optimization packages, like `mealpy` is under development.
    (Maybe include more `io` for global optimization algorithms like `COA`, `GWO` and etc.)  
  Currently, we are testing the effciency of `GWO`, and in my further research, more global optimization methods  
    will be applied and restored in this library.

## Importance

  Here the `asop` method is just a copyright version by [Filend](https://github.com/Fil-end), you may find the official edition from [ZhiPan Liu Group](https://zpliu.fudan.edu.cn/).
