#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plotfile = sys.argv[1]
modelname = sys.argv[2]

ytsvfile =  sys.argv[3] # "iteration.tsv"
ctsvfile =  sys.argv[4] # "checkpoint.tsv"
etsvfile =  sys.argv[5] # "eval.tsv" - Not used as no training iterations number
stsvfile =  sys.argv[6] # "sub.tsv"
ltsvfile =  sys.argv[7] # "lstmeval.tsv"

maxticks=10

# Read and sort data files more efficiently - chain operations
ydf = pd.read_csv(ytsvfile, sep='\t', encoding='utf-8').sort_values('TrainingIteration')
cdf = pd.read_csv(ctsvfile, sep='\t', encoding='utf-8').sort_values('TrainingIteration')
sdf = pd.read_csv(stsvfile, sep='\t', encoding='utf-8').sort_values('TrainingIteration')
ldf = pd.read_csv(ltsvfile, sep='\t', encoding='utf-8').sort_values('TrainingIteration')

# Extract columns once as numpy arrays to avoid repeated dataframe access
y = ydf['IterationCER'].values
x = ydf['LearningIteration'].values
t = ydf['TrainingIteration'].values

c = cdf['CheckpointCER'].values
cx = cdf['LearningIteration'].values
ct = cdf['TrainingIteration'].values

s = sdf['SubtrainerCER'].values
sx = sdf['LearningIteration'].values
st = sdf['TrainingIteration'].values

l = ldf['EvalCER'].values
lx = ldf['LearningIteration'].values
lt = ldf['TrainingIteration'].values

def annot_min(boxcolor, xpos, ypos, x, y, z):
    # Use numpy for efficient min finding on arrays
    valid_indices = ~np.isnan(y)
    if not valid_indices.any():
        return
    
    valid_y = y[valid_indices]
    valid_x = x[valid_indices]
    valid_z = z[valid_indices]
    
    min_idx = np.argmin(valid_y)
    tmin = valid_z[min_idx]
    xmin = valid_x[min_idx]
    ymin = valid_y[min_idx]
    boxtext = " {:.3f}% at {:,} / {:,} " .format(ymin, xmin, tmin)
    ax1.annotate(boxtext, xy=(tmin, ymin), xytext=(xpos, ypos), textcoords='offset points', color='black', fontsize=9,
        arrowprops=dict(shrinkA=1, shrinkB=1, fc=boxcolor, alpha=0.7, ec='white', connectionstyle="arc3"),
        bbox=dict(boxstyle='round,pad=0.2', fc=boxcolor, alpha=0.3))

PlotTitle="Tesseract LSTM Training : " + modelname
fig = plt.figure(figsize=(11,8.5)) #size is in inches
ax1 = fig.add_subplot()

ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
ax1.set_ylabel('Error Rate %')

ax1.set_xlabel('Training Iterations')
ax1.set_xticks(t)
ax1.tick_params(axis='x', labelsize='small')
ax1.locator_params(axis='x', nbins=maxticks)  # limit ticks on x-axis
ax1.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

ax1.scatter(t, y, c='teal', alpha=0.7, s=0.5, label='BCER at #iterations/100 - lstmtraining - list.train')
ax1.plot(t, y, 'teal', alpha=0.3, linewidth=0.5, label='Training BCER')
ax1.grid(True)

# Check for non-NaN values more efficiently using numpy
c_valid = ~np.isnan(c).all()
if c_valid:
    ax1.scatter(ct, c, c='teal', marker='x', s=35,
       label='BCER at checkpoints - lstmtraining - list.train', alpha=0.5)
    annot_min('teal',0,-50,cx,c,ct)

l_valid = ~np.isnan(l).all()
if l_valid:
    ax1.plot(lt, l, 'magenta', linewidth=0.5, label='Validation BCER')
    ax1.scatter(lt, l, c='magenta', s=10, label='BCER at checkpoints - lstmeval - list.eval', alpha=0.5)
    annot_min('magenta',0,-50,lx,l,lt)

s_valid = ~np.isnan(s).all()
if s_valid:
    ax1.plot(st, s, 'orange', linewidth=0.5, label='SubTrainer BCER')
    ax1.scatter(st, s, c='orange', s=0.5,
       label='BCER for UpdateSubtrainer every 100 iterations', alpha=0.5)
    annot_min('orange',-100,-100,sx,s,st)

tmax = t[np.argmax(x)]
ymax = y[np.argmax(x)]
xmax = x.max()
boxtext= " {:.3f}% at \n  {:,} \n {:,} " .format(ymax,xmax,tmax)
ax1.annotate(boxtext, xy=(tmax, ymax), xytext=(20,-10), textcoords='offset points', color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='teal', alpha=0.3))

plt.title('character error rate over training iterations - from lstmtraining and lstmeval',fontsize=10)
plt.suptitle(PlotTitle, y=0.95, fontsize = 14, fontweight = 'bold')
plt.legend(loc='upper right')

ax1.set_ylim([-0.5,100])

# Secondary x axis on top to display Learning Iterations
ax2 = ax1.twiny() # ax1 and ax2 share y-axis
ax2.set_xlabel("Learning Iterations")
ax2.set_xlim(ax1.get_xlim()) # ensure the independent x-axes now span the same range
ax2.set_xticks(t) # copy over the locations of the x-ticks from Training Iterations
ax2.tick_params(axis='x', labelsize='small')
ax2.set_xticklabels(matplotlib.ticker.StrMethodFormatter('{x:,.0f}').format_ticks(x)) # But give value of Learning Iterations
ax2.locator_params(axis='x', nbins=maxticks)  # limit ticks to same as x-axis
ax2.xaxis.set_ticks_position('bottom') # set the position of ticks of the second x-axis to bottom
ax2.xaxis.set_label_position('bottom') # set the position of labels of the second x-axis to bottom
ax2.spines['bottom'].set_position(('outward', 36)) # positions the second x-axis below the first x-axis

plt.savefig(plotfile)
