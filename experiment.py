from xml.etree.ElementInclude import default_loader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from argparse import ArgumentParser
#import wandb

from model import Net


def main(args):
    # Dataset
    x = torch.linspace(-1,1,steps=args.num_data).view(-1,1)
    y = torch.ones(args.num_data,1)
    y[torch.arange(args.num_data)%2==0] = 0
    #y = torch.randn(args.num_data).view(-1,1)

    xx = torch.linspace(-2, 2, 1000).view(-1,1)
    checkpoints = (2 ** np.arange(150)).astype(int)
    checkpoints = checkpoints[checkpoints>0]
    checkpoints = [0] + sorted(set(checkpoints[checkpoints<args.num_iter])) + [args.num_iter-1]
    #checkpoints = range(0, args.num_iter, 1e8)
    #checkpoints = [0, args.num_iter-1]
    print(checkpoints)

    loss = nn.MSELoss()

    #if args.wandb_log:
    #    wandb.init(project='AI616project', config=args)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    print("Seed:", torch.initial_seed())
    
    #try:
    if True:
        model = Net(width=args.width, fix_w1=args.fix_w1, scale=args.scale)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.001)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        #optimizer = optim.Adam(model.parameters(), lr=args.lr)
        #optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        #optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        #model.print_weights()

        if args.print_plots:
            xscale = 'log' if args.log_scale_plot else 'linear'
            with torch.no_grad():
                yy0 = model(xx).numpy()
                weight1 = model.fc1.weight.data.view(-1,1).numpy()
                bias1 = model.fc1.bias.data.view(-1,1).numpy()
                weight2 = model.fc2.weight.data.view(-1,1).numpy()
            if args.print_knots:
                knots = model.get_knots().view(-1,1)
                timestamp = [0]
                x_knots = knots.view(-1,1).numpy()
                with torch.no_grad():
                    y_knots = model(knots).view(-1,1).numpy()
        
        pbar = tqdm(range(args.num_iter))
        costs = []
        colors = plt.cm.rainbow(np.linspace(0, 1, args.width))
        for i in pbar:
            
            model.train()
            out = model(x)
            cost = loss(out, y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            costs.append(float(cost.item()))

            if i in checkpoints or i % 1000 == 0:
                pbar.set_description(f"Step {i}: Cost {cost.item():.8f}")

            if args.print_knots and (i in checkpoints or i % args.knots_gap == 0):
                timestamp.append(i+1)
                knots = model.get_knots().view(-1,1)
                x_knots = np.concatenate([x_knots, knots.view(-1,1).numpy()], axis=1)
                with torch.no_grad():
                    y_knots = np.concatenate([y_knots, model(knots).view(-1,1).numpy()], axis=1)
                    weight1_ = model.fc1.weight.data.view(-1,1).numpy()
                    bias1_ = model.fc1.bias.data.view(-1,1).numpy()
                    weight2_ = model.fc2.weight.data.view(-1,1).numpy()
                    weight1 = np.concatenate([weight1, weight1_], axis=1)
                    bias1 = np.concatenate([bias1, bias1_], axis=1)
                    weight2 = np.concatenate([weight2, weight2_], axis=1)
            
            if i in checkpoints:
                #model.print_weights()                

                #if args.wandb_log:
                #    wandb.log({'cost': cost.item()}, step=i)
                
                if args.print_plots:
                    model.eval()

                    # plot
                    plt.subplots(figsize=(16, 9), constrained_layout=True)
                    plt.subplot(2,3,1)
                    plt.plot(xx.numpy(), yy0, '--', color='0.7', linewidth=0.7, label='initialization')
                    plt.scatter(x,y, s=100, color=(1,0,0.5),marker='*', label='data points')

                    #Linear regression
                    #w_lin = 0 if args.num_data%2 else 3/(2*args.num_data)
                    #b_lin = 0.5 * (args.num_data-1)/args.num_data if args.num_data%2 else 0.5
                    #x_lin = np.array([-10, 10])
                    #y_lin = w_lin * x_lin + b_lin
                    #plt.plot(x_lin, y_lin, '-.', color='0.5', linewidth=.7, label='least square line')
                    

                    with torch.set_grad_enabled(False):
                        yy = model(xx).numpy()

                    plt.plot(xx.numpy(), yy, c='blue', linewidth=1, label='Neural Net $f_w (x)$')

                    if args.print_knots:
                        for i_, (x_knot, y_knot) in enumerate(zip(x_knots, y_knots)):
                            #plt.scatter(x_knot, y_knot, s=0.2, marker='.')
                            plt.plot(x_knot, y_knot, linewidth=0.7, marker='o', markersize=1, color=colors[i_]) 

                    plt.xlim(-1.5,1.5)
                    plt.ylim(-1.5, 2.5)
                    plt.grid(True, which='major',linewidth=0.2)
                    plt.legend(ncol=2)
                    plt.title('Plot of NN')
                    plt.xlabel('x')
                    plt.ylabel('y')

                    # knot x 좌표
                    if args.print_knots:
                        plt.subplot(2,3,2)
                        for i_, x_knot in enumerate(x_knots):
                            plt.plot(timestamp, x_knot, marker='o', markersize=1.5, color=colors[i_])
                            #plt.annotate(str(i_), (timestamp[-1], x_knot[-1]), fontsize=10)
                        plt.hlines(x, 0, timestamp[-1]+1, linestyles='dashed', color='black', label='x-coord. of data points')
                        plt.ylim(-1.5, 1.5)
                        plt.xscale(xscale)
                        plt.legend()
                        plt.title('$x$-coordinates of knots')
                        plt.xlabel('time (log scale)')
                        plt.ylabel('position of knots')
                        
                        # loss 
                        plt.subplot(2,3,3)
                        plt.plot(costs)
                        plt.xscale(xscale)
                        plt.title('Square Loss')
                        plt.xlabel('time (log scale)')
                        plt.ylabel('Loss')
                        #plt.show(block=False)
                    
                    # weights
                    if True:
                        plt.subplot(2,3,4)
                        for i_, w in enumerate(weight1):
                            plt.plot(timestamp, w, marker='o', markersize=1, color=colors[i_])
                        plt.xscale(xscale)
                        plt.title("Weight of the first layer ($a_i$'s)")
                        plt.xlabel('time (log scale)')
                        plt.ylabel("$a_i$")
                        plt.subplot(2,3,5)
                        for i_, b in enumerate(bias1):
                            plt.plot(timestamp, b, marker='o', markersize=1, color=colors[i_])
                        plt.xscale(xscale)
                        plt.title("Bias of the first layer ($b_i$'s)")
                        plt.xlabel('time (log scale)')
                        plt.ylabel("$b_i$")
                        plt.subplot(2,3,6)
                        for i_, v in enumerate(weight2):
                            plt.plot(timestamp, v, marker='o', markersize=1, color=colors[i_])
                        plt.xscale(xscale)
                        plt.title("Weight of the second layer ($c_i$'s)")
                        plt.xlabel('time (log scale)')
                        plt.ylabel("$c_i$")
                        plt.show(block=False)
                        
                    input("Press Enter key...")
                    #sleep(10)
                    plt.close('all')
    #except Exception as e:
    #    print(e)
    #    if args.wandb_log:
    #        wandb.finish()
    #    sys.exit(1)
    
    #if args.wandb_log:
    #    wandb.finish()

if __name__ == '__main__':
    #np.set_printoptions(4)
    args = ArgumentParser("AI616 Project: Univariate Two-Layer ReLU Network")

    args.add_argument('-m', '--num-data', default=9, type=int, help="Number of data points in the range [-1,1]. (default: 9)")
    args.add_argument('-i', '--num-iter', default=int(1e8), type=int, help="Number of iterations. (default: 1e8)")
    args.add_argument('-n', '--width', default=100, type=int, help='Width, or the number of ReLU neurons, of the network. (default: 100)')
    args.add_argument('-l', '--lr', default=1e-3, type=float, help="Learning Rate")
    args.add_argument('-s', '--seed', default=None, type=int, help="For reproducibility")
    args.add_argument('-g', '--knots-gap', default=1000, type=int, help='gap between knot printed')
    args.add_argument('-S', '--scale', default=1., type=float, help='multiplying the first layer by the given value, while dividing the second layer by the given value')

    args.add_argument('-f', '--fix-w1', action='store_true', help='If you use this option, the weights(not bias) of the first layer get fixed by all-1-vector')
    args.add_argument('-p', '--print-plots', action='store_true', help='If you use this option, the plots will appear')
    args.add_argument('-k', '--print-knots', action='store_true', help='If you use this option, the position of knots will appear on a plot')
    args.add_argument('-L', '--log-scale-plot', action='store_true', help='If you use  this option, most of plots are in logscale (time axes)')
    #args.add_argument('-w', '--wandb-log', action='store_true', help='If you want to use WandB, use this option')
    

    main(args.parse_args())