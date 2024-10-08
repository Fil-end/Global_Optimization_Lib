# Global_Optimization_Lib

The `global_optimization` package is a Python library designed for restoration of global optimization algorithms.

## Currently restored:
​	`ga`: `src/ga/ga.py`  
  `ssw`: `src/ssw/main.py`  
  `asop`: `src/asop/asop.py`  
  
## Tips
  Here we bind `calculator_method:LASP` to `SSW` while `calculator_method: MACE` to `GA`.  
    (Currently, our main choice in `asop` is `calculator_method: LASP` to `SSW`, because `SSW` is a much more efficient global optimization algorithm than `GA`).  
    
​	If necessary, this library will include more global optimization methods for the following research.  
    (Maybe include more `io` for global optimization algorithms like `COA`, `GWO` and etc.)  

## Importance
  Here the `asop` method is just a copyright version by [Filend](https://github.com/Fil-end), you may find the official edition from [ZhiPan Liu Group](https://zpliu.fudan.edu.cn/).
