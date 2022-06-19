# Shallow-Univariate-ReLU-Net

`-m`, `--num-data`: Number of data points in the range [-1,1]. (`default=1e8, type=int`)  
`-i`, `--width`: Width, or the number of ReLU neurons, of the network. (`default=100, type=int`)  
`-l`, `--lr`: Learning rate. (`default=1e-3, type=float`)  
`-s`, `--seed`: For reproducibility. (`default=None, type=int`)  
`-g`, `--knots-gap`: Gap of timesteps between printing knots. (`default=1000, type=int`)  
`-S`, `--scale`: Multiplying the first layer by the given value, while dividing the second layer by the given value. (`default=1., type=float`)  

`-f`, `--fix-w1`, If you use this option, the weights(not bias) of the first layer get fixed by all-1-vector.  
`-p`, `--print-plots`, If you use this option, the plots will appear. (At every `t==2**n`)  
`-k`, `--print-knots`, If you use this option, the position of knots will appear on a plot.  
`-L`, `--log-scale-plot`, If you use  this option, most of plots are in logscale (time axes)  

## Example run

- Data size 5, Width 10, LR $10^{-3}$, Total timesteps $10^{7}$, Printing the position of knots at every 1 timestep(s), Fix weight at 1st layer, Log scale time axes: 
```
python3 experiment.py -m 5 -n 10 -l 1e-3 -i 10000000 -g 1 -fpkL
```