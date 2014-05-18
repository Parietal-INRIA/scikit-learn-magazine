import urllib, os # Standard library imports

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, ensemble

###############################################################################
# Load the data
import pandas

if not os.path.exists('wages.txt'):
    urllib.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages',
                       'wages.txt')

names = [
    'EDUCATION: Number of years of education',
    'SOUTH: 1=Person lives in South, 0=Person lives elsewhere',
    'SEX: 1=Female, 0=Male',
    'EXPERIENCE: Number of years of work experience',
    'UNION: (1=Union member, 0=Not union member)',
    'WAGE: Wage (dollars per hour)',
    'AGE: years',
    'RACE: 1=Other, 2=Hispanic, 3=White',
    'OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other',
    'SECTOR: 0=Other, 1=Manufacturing, 2=Construction',
    'MARR: 0=Unmarried,  1=Married',
]

short_names = [n.split(':')[0] for n in names]

data = pandas.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None,
       header=None)
data.columns = short_names

# In the color plots, crop at 30 to have more dynamical range on the
# color
wage_max = 30

for name, learner in [('', None),
                      (' linear SVM', svm.SVR(kernel='linear')),
                      (' random forest', ensemble.RandomForestRegressor())]:
    #-----------------------------------------------------------------------
    # Plot a simple pair-wise plot
    plt.figure(figsize=(6, 5))

    ax_2d = plt.axes([.35, .33, .6, .6])

    # Add random offset to limit overlap in points
    n_sample = len(data)
    plt.scatter(data['EXPERIENCE'] -.25 + .5 * np.random.random(n_sample),
                data['EDUCATION'] -.25 + .5 * np.random.random(n_sample),
                c=data['WAGE'], cmap=plt.cm.Blues, vmax=wage_max)

    plt.xlabel('Years of work experience')
    plt.ylabel('Years of education')
    plt.yticks(size=10)
    plt.xticks(size=10)

    plt.axis('tight')
    ex_min, ex_max = plt.xlim()
    ed_min, ed_max = plt.ylim()
    cb = plt.colorbar()
    # Squeeze the colorbar a bit left
    bb = cb.ax.get_position()
    bb.x0 -= .03
    cb.ax.set_position(bb)
    cb.ax.set_ylabel('Wage')
    for l in cb.ax.get_yticklabels():
        l.update(dict(size=10))

    # Side plot as a function of education
    ax_ed = plt.axes([.11, .33, .2, .6])
    plt.plot(data['WAGE'], data['EDUCATION'], '+', label='Observed\n data')
    plt.ylim(ed_min, ed_max)
    plt.xlim(xmax=48)
    plt.ylabel('Years of education')
    plt.xlabel('Wage')
    plt.xticks((10, 20, 30, 40), size=10)
    plt.yticks(size=10)

    # Side plot as a function of experience
    ax_ex = plt.axes([.35, .09, .48, .2])
    plt.plot(data['EXPERIENCE'], data['WAGE'], '+', label='Observed\n data')
    plt.ylabel('Wage')
    plt.yticks((10, 20, 30, 40), size=10)
    plt.xticks(size=10)
    plt.xlabel('Years of experience')
    plt.xlim(ex_min, ex_max)
    plt.ylim(ymax=48)

    plt.suptitle('Wage in function of years of education and experience     .',
                 size=14)

    #-----------------------------------------------------------------------
    # Some learning
    if learner is not None:
        # On the 2D plot
        learner.fit(np.array((data['EXPERIENCE'], data['EDUCATION'])).T,
                    data['WAGE'])

        grid = np.mgrid[ex_min - 1:ex_max + 1:100j, ed_min - 1:ed_max + 1:100j]
        prediction = learner.predict(grid.reshape((2, -1)).T)
        prediction = np.reshape(prediction, (100, 100))

        ax_2d.set_autoscale_on(False)
        ax_2d.imshow(np.rot90(prediction), vmin=data['WAGE'].min(),
                    vmax=wage_max, aspect='auto',
                    extent=(ex_min - 1, ex_max, ed_min -1, ed_max + 1),
                    cmap=plt.cm.Reds)

        # On the education plot
        learner.fit(data['EDUCATION'][:, np.newaxis], data['WAGE'])

        grid = np.mgrid[ed_min - 1:ed_max + 1:100j]
        prediction = learner.predict(grid[:, np.newaxis])

        ax_ed.set_autoscale_on(False)
        ax_ed.plot(prediction, grid, 'r', lw=2)

        # On the experience plot
        learner.fit(data['EXPERIENCE'][:, np.newaxis], data['WAGE'])

        grid = np.mgrid[ex_min - 1:ex_max + 1:100j]
        prediction = learner.predict(grid[:, np.newaxis])

        ax_ex.set_autoscale_on(False)
        ax_ex.plot(grid, prediction, 'r', label='Prediction:\n%s' % name,
                   lw=2)

        ax_ex.legend(loc=(-.74, -.22), prop=dict(size=12), frameon=False,
                     handletextpad=.3, labelspacing=1)
        plt.suptitle('Wage:%s prediction' % name, size=14)

    plt.savefig('wage_data%s.pdf' % name.lower().replace(' ', '_'))
