# Global_Optimization_Lib

  &emsp;The `global_optimization` package is a Python library designed for restoration of global optimization algorithms. 
  &emsp;Currently, the io between `ase.Atoms` and `mealpy` project (a library which restores comprehensive global optimization  
  algorithms) has been preliminarily added into this project for further research.

## Currently restored

​ `ga`: `src/ga/ga.py`  
  `ssw`: `src/ssw/main.py`  
  `asop`: `src/asop/asop.py`  
  `gwo`: `src/gwo/GWO.py`  
  `mealpy`: `src/mealpy_io`
  
## Tips

  &emsp;Here we bind `calculator_method:LASP` to `SSW` while `calculator_method: MACE` to `GA`.  
    (Currently, our main choice in `asop` is `calculator_method: LASP` to `SSW`, because `SSW` is a much more efficient global optimization algorithm than `GA`).

## Developing  

  &emsp;Here, the I/O between `ase.Atoms` and global optimization packages, like `mealpy` is under development.
    (Maybe include more `io` for global optimization algorithms like `COA`, `GWO` and etc.)  
  &emsp;Currently, we are testing the effciency of `GWO`, and in my further research, more global optimization methods  
    will be applied and restored in this library.

## Importance

  Here the `asop` method is just a copyright version by [Filend](https://github.com/Fil-end), you may find the official edition from [ZhiPan Liu Group](https://zpliu.fudan.edu.cn/).
