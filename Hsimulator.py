"""
    file      : Hsimulator.py
    author    : Hasanuddin
    copyright : 2021
    purpose   : planet simulator using leapfrog KDK
"""

from math import sqrt,ceil
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import time as CPUtime

class orbit():
    '''
        class representing orbit
		x     :  position      (numpy 3x1)
        v     :  velocity      (numpy 3x1)
        p     :  potential     (scalar)
        a     :  acceleration  (numpy 3x1)
    '''
    def __init__(self, x, v, p=0,a=0):
        self.x = x
        self.v = v
        self.p = p
        self.a = a

    def set_gravity(self, pot, acc):
        self.p = pot
        self.a = acc

    def kick(self,dt):
        self.v = self.v + (self.a*dt)

    def drift(self,dt):
        self.x = self.x + (self.v*dt)

class snapshot():
    '''
        snapshot orbits
    '''
    def __init__(self, time, orbits):
        self.time   = time
        self.orbits = orbits
    def __str__(self):
        return ("snapshot at time = "+str(self.time)+" : "+str(len(self.orbits))
		        +" orbit(s)")

class potential():
    '''
        class representing point mass potential
    '''
    def __init__(self, G=1,M=1):
        self.GM = G*M

    def pot(self,x):
        rq = x.dot(x)
        r = sqrt(rq)
        return 	-self.GM/r

    def acc(self,x):
        rq = x.dot(x)
        r = sqrt(rq)
        return -self.GM*x/r/rq

def print_header():
    print(" ")
	#rint('1234567890123456789012345678901234567890123456789012345678901234567')
    print("*******************************************************************")
    print("*                      PLANET SIMULATION                           ")
    print("*******************************************************************")
    print("* Author    : Hasanuddin ")
    print("* copyright : 2021       ")
    print("* version   : 1.0.0      ")
    print("*******************************************************************")
    print(" ")

def run(snap, tf = 100, dt=0.01, method='leapfrog',gravity = 'pointmass',
        G=1, M=1, savefile=' ', snapfile=' ', snapdt=0):
    '''
        param        description                   param type
        ------------------------------------------------------
	    snap     : snapshot of orbit                (snapshot)
        ti       : initial time                     (scalar)
        tf       : final   time                     (scalar)
        dt       : step size                        (scalar)
        method   : integrator ['leapfrog']          (string)
        gravity  : potential  ['pointmass']         (string)
        G        : G [1]                            (scalar)
        M        : Mass of planet [1]               (scalar)
        savefile : output file [' ']
                   contains t x v p a  1st orbit    (string)
        snapfile : snapshot file  [' ']
                   at the end simulation            (string)
        snapdt   : interval snapfile save [0]       (scalar)
    '''
    print_header()
	# remove snapfile extension
    idot = snapfile.find('.')
    if idot>-1:
        snapfileWOExten=snapfile[:idot]
    else:
        snapfileWOExten=snapfile
	#potential
    P = potential(G,M)
    print("Potential set for pointmass with G = ", str(G)," & M = ",str(M))
    print(" ")

    print("Integrator set for " , str(method)," with step size = ",str(dt))
    print(" ")
	#Read snap
    Norbit = len(snap.orbits)
    ti = snap.time
    print("Reading snapshot ... ")
    assert Norbit, " No orbit found"
    print(" ",str(Norbit)," orbit(s) found")
    print(" ")

    #Trace First Orbit
    FirstOrb = snap.orbits[0]
    Pos  = [FirstOrb.x ]
    Vel  = [FirstOrb.v ]
    Pot  = [P.pot(FirstOrb.x)]
    Time = [ti]

    #run information
    RunInfoStr = ("run(snap="+str(snap)+", tf = " +str(tf)+", dt= "+str(dt) + ", method="+str(method)+",gravity = "+str(gravity)+", G = "+str(G)+", M = "+
    str(M)+", savefile= "+str(savefile)+", snapfile= "+str(snapfile) +", snapdt = "+str(snapdt)+")")

    #start
    CPUstart = CPUtime.time()
    t = ti
    dth = 0.5*dt
    step = 0

    # 0 -> 100%
    didel = "\b"*3   # backspace 3 times
    tsim = tf-ti
    print("Simulation in progress ... ")
    while t<tf:
        for O in snap.orbits:
            if method=='leapfrog':
                # calculate pot and acc
                O.set_gravity(P.pot(O.x),P.acc(O.x))
                # KICK HALF
                O.kick(dth)
                # DRIFT FULL
                O.drift(dt)
                # calculate pot and acc again
                O.set_gravity(P.pot(O.x),P.acc(O.x))
                # KICK HALF AGAIN
                O.kick(dth)
            elif method=='heun':
                O.set_gravity(P.pot(O.x), P.acc(O.x))
                k1 = dt*P.acc(O.x)
                l1 = dt*O.v
                k2 = dt*P.acc(O.x+l1)
                l2 = dt*(O.v+k1)
                O.x = O.x + 0.5*(l1+l2)
                O.v = O.v + 0.5*(k1+k2)
                O.set_gravity(P.pot(O.x), P.acc(O.x))
            else: #method=='RK3'
                O.set_gravity(P.pot(O.x),P.acc(O.x))
                k1 = dt*P.acc(O.x)
                l1 = dt*O.v
                k2 = dt*P.acc(O.x+0.5*l1)
                l2 = dt*(O.v+0.5*k1)
                k3 = dt*P.acc(O.x-l1+2*l2)
                l3 = dt*(O.v -k1+2*k2)
                O.x = O.x + (l1+4*l2+l3)/6.
                O.v = O.v + (k1+4*k2+k3)/6.
                O.set_gravity(P.pot(O.x), P.acc(O.x))
        # change time
        t += dt
        step += 1
        snap.time = t
        # fill trace and time
        Pos.append(snap.orbits[0].x )
        Vel.append(snap.orbits[0].v )
        Pot.append(P.pot(snap.orbits[0].x))
        Time.append(t)
        if snapdt:
            nt = step*dt/snapdt
        if snapdt and (ceil(nt)==nt):
            data = [snap, RunInfoStr]
            with open(snapfileWOExten+str(ceil(nt))+'.pkl','wb') as file:
                pickle.dump(data,file)
        '''
        print("%9.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
		     %(snap.time,snap.orbits[0].x[0], snap.orbits[0].x[1],
			 snap.orbits[0].x[2], snap.orbits[0].v[0], snap.orbits[0].v[1],
			 snap.orbits[0].v[2] ))
        '''
        progress = int((t-ti)/tsim*100)
        print("{0}{1:{2}}".format(didel, progress,3), end="")
        sys.stdout.flush()

    print(" ")
	# steps information:
    print('Number of steps = '+str(step))
    print(" ")


    if savefile != ' ' :
        #saving first orbit data to pickle
        data = [Time, Pos, Vel, Pot, RunInfoStr]
        # Time -> scalar ,  Pos -> vector , Vel -> Vector, Pot -> Scalar
        with open(savefile,'wb') as file:
            pickle.dump(data,file)

    if snapfile != ' ' :
        #saving last snap file
        data = [snap, RunInfoStr]
        with open(snapfile,'wb') as file:
            pickle.dump(data,file)

    # CPU time information
    print("Execution time %.3f second"%(CPUtime.time()-CPUstart))
    print(" ")
    X = [posi[0] for posi in Pos]
    Y = [posi[1] for posi in Pos]
    plt.axes().set_aspect('equal')
    plt.plot(X,Y)
    plt.scatter(0,0,c='orange',s=64)
    plt.scatter(X[-1],Y[-1],c='black')
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')

    plt.show()


def init_snap(xs,vs,t=0):
    '''
        make initial snapshot
        xs = [xi] list of positions
    	vs = [vi] list of velocity
		t =  snapshot time
    '''
    kepler = potential()
    N = len(xs)
    #assert (N=len(vs)),"length of xs : "+str(N)+" != length of vs : "+str(len(vs))
    orb = []
    for i in range (N):
        x = np.array(xs[i])
        v = np.array(vs[i])
        p = kepler.pot(x)
        a = kepler.acc(x)
        o = orbit(x,v,p,a)
        orb.append(o)
    snap =snapshot(t,orb)
    return snap

def init():
    '''
        make initial snapshot sample
    '''
    kepler = potential()
    x0 = np.array([1.00,0.00,0.00])
    v0 = np.array([0.00,0.50,0.00])
    p0 = kepler.pot(x0)
    a0 = kepler.acc(x0)

    x1 = np.array([1.01,0.00,0.00])
    v1 = np.array([0.00,0.51,0.00])
    p1 = kepler.pot(x1)
    a1 = kepler.acc(x1)

    x2 = np.array([1.01,0.01,0.00])
    v2 = np.array([0.01,0.51,0.00])
    p2 = kepler.pot(x2)
    a2 = kepler.acc(x2)

    x3 = np.array([1.00,0.01,0.00])
    v3 = np.array([0.01,0.50,0.00])
    p3 = kepler.pot(x3)
    a3 = kepler.acc(x3)

    x4 = np.array([0.99,0.01,0.00])
    v4 = np.array([0.01,0.49,0.00])
    p4 = kepler.pot(x4)
    a4 = kepler.acc(x4)

    x5 = np.array([0.99,0.00,0.00])
    v5 = np.array([0.00,0.49,0.00])
    p5 = kepler.pot(x5)
    a5 = kepler.acc(x5)

    x6 = np.array([0.99,-0.01,0.00])
    v6 = np.array([-0.01,0.49,0.00])
    p6 = kepler.pot(x6)
    a6 = kepler.acc(x6)

    x7 = np.array([1.00,-0.01,0.00])
    v7 = np.array([-0.01,0.50,0.00])
    p7 = kepler.pot(x7)
    a7 = kepler.acc(x7)

    x8 = np.array([1.01,-0.01,0.00])
    v8 = np.array([-0.01,0.51,0.00])
    p8 = kepler.pot(x8)
    a8 = kepler.acc(x8)


    o0 = orbit(x0,v0,p0,a0)
    o1 = orbit(x1,v1,p1,a1)
    o2 = orbit(x2,v2,p2,a2)
    o3 = orbit(x3,v3,p3,a3)
    o4 = orbit(x4,v4,p4,a4)
    o5 = orbit(x5,v5,p5,a5)
    o6 = orbit(x6,v6,p6,a6)
    o7 = orbit(x7,v7,p7,a7)
    o8 = orbit(x8,v8,p8,a8)

    snap =snapshot(0,[o0,o1,o2,o3,o4,o5,o6,o7,o8])
    return snap

def plot_XY(fname,otherfile=' ', xlim=[0,0],ylim=[0,0], savefig=' ',dpi=300
    , plot_xaxis=False, plot_yaxis=False, legend_label=' ', legend_loc=-1,legend_label2=' '):
    #Name:
    #    plot_XY
    """simply plot orbit

    parameters
    ----------
    fname : str
        name of file containing t,x,v.
    otherfile : str
	    name of another file containing t,x,v
    xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.
    dpi  : int, optional
        dots per inch
    plot_xaxis : bool, optional
        do you want to plot x-axis (y=0)?
    plot_yaxis : bool, optional
        do you want to plot y-axis (x=0)?
    legend_label: str, optional
        label for legend
    legend_loc: int, optional
        location of legend
    legend_label2: str, optional
        label2 for legend


    returns
    -------
    none
        none if savefig not given.
    figure: file
        show figure and save figure if savefig given.


    |
    """

    #History
    # 02-11-2021  written Hasanuddin

    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    #time = data[0]
    pos = data[1]
    #print (pos)
    #vel = data[2]
    x = [xt[0] for xt in pos]
    y = [xt[1] for xt in pos]

    if otherfile !=' ':
        with open(otherfile,'rb') as file:
            data = pickle.load(file)
            pos2 = data[1]
            vel2 = data[2]
            x2 = [xt[0] for xt in pos2]
            y2 = [xt[1] for xt in pos2]

    plt.plot(x, y , 'r',label=legend_label)
    if otherfile !=' ':
        plt.plot(x2,y2, 'b',label=legend_label2)
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y$',fontsize=16)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if plot_xaxis:
        plt.hlines(0,xlim[0],xlim[1],color='k')
    if plot_yaxis:
        plt.vlines(0,ylim[0],ylim[1],color='k')
    #add legend
    if legend_loc!=-1:
        plt.legend(loc=legend_loc)
    # same aspect axis
    plt.axes().set_aspect('equal')
    # tight layout
    plt.tight_layout()
    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)
    plt.show()


def plot_phase_space(fname,otherfile=' ', xlim=[0,0],ylim=[0,0], savefig=' ',
    dpi=300, plot_xaxis=False, plot_yaxis=False, legend_label=' ', legend_loc=-1,legend_label2=' ',plot_phase_space=True):
    #Name:
    #    plot_phase_space
    """simply plot phase space

    parameters
    ----------
    fname : str
        name of file containing t,x,v.
    otherfile : str
	    name of another file containing t,x,v
    xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.
    dpi  : int, optional
        dots per inch
    plot_xaxis : bool, optional
        do you want to plot x-axis (y=0)?
    plot_yaxis : bool, optional
        do you want to plot y-axis (x=0)?
    legend_label: str, optional
        label for legend
    legend_loc: int, optional
        location of legend
    legend_label2: str, optional
        label2 for legend
    plot_phase_space: bool, optional
        plot_phase_space?

    returns
    -------
    none
        none if savefig not given.
    figure: file
        show figure and save figure if savefig given.


    |
    """

    #History
    # 23-11-2021  written Hasanuddin

    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    #time = data[0]
    pos = data[1]
    #print (pos)
    vel = data[2]
    x = [xt[0] for xt in pos]
    y = [xt[1] for xt in pos]
    vx = [vt[0] for vt in vel]
    vy = [vt[1] for vt in vel]

    if otherfile !=' ':
        with open(otherfile,'rb') as file:
            data = pickle.load(file)
            pos2 = data[1]
            vel2 = data[2]
            x2 = [xt[0] for xt in pos2]
            y2 = [xt[1] for xt in pos2]
            vx2 = [vt[0] for vt in vel2]
            vy2 = [vt[1] for vt in vel2]

    plt.figure(1)
    plt.plot(x, vx , 'r',label=legend_label)
    if otherfile !=' ':
        plt.plot(x2,vx2, 'b',label=legend_label2)
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$v_x$',fontsize=16)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if plot_xaxis:
        plt.hlines(0,xlim[0],xlim[1],color='k')
    if plot_yaxis:
        plt.vlines(0,ylim[0],ylim[1],color='k')
    #add legend
    if legend_loc!=-1:
        plt.legend(loc=legend_loc)
    # same aspect axis
    #plt.axes().set_aspect('equal')
    # tight layout
    plt.tight_layout()
    if savefig!=' ' :
        plt.savefig(savefig+str(1)+'.jpg',dpi=dpi)

    plt.figure(2)
    plt.plot(y, vy , 'r',label=legend_label)
    if otherfile !=' ':
        plt.plot(y2,vy2, 'b',label=legend_label2)
    plt.xlabel(r'$y$',fontsize=16)
    plt.ylabel(r'$v_y$',fontsize=16)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if plot_xaxis:
        plt.hlines(0,xlim[0],xlim[1],color='k')
    if plot_yaxis:
        plt.vlines(0,ylim[0],ylim[1],color='k')
    #add legend
    if legend_loc!=-1:
        plt.legend(loc=legend_loc)
    # same aspect axis
    #plt.axes().set_aspect('equal')
    # tight layout
    plt.tight_layout()
    if savefig!=' ' :
        plt.savefig(savefig+str(2)+'.jpg',dpi=dpi)

    plt.show()


def plot_radius_time(fname, xlim=[0,0],ylim=[0,0], savefig=' '):
    """
        print radius ve time
    """
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = data[0]
    pos = data[1]

    r = np.array([sqrt(xi.dot(xi)) for xi in pos])
    plt.plot(time,r)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$r$')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)
    plt.show()

def plot_energy_time(fname, otherfile=' ', type='de', xlim=[0,0],ylim=[0,0], savefig=' ',dpi=300, period=1.,plot_xaxis = False,legend_label=' ',
legend_loc=-1,legend_label2=' ',tight_layout = True):
    """ plot energy('E') :math:`E` or error ('dE') :math:`\Delta E`,
	    relative energy error ('de') :math:`\Delta E / E_0` over time.

    parameters
    ----------
    fname : str
        name of file containing t,x,v.
	otherfile : str
	    name of another file containing t,x,v
    type : str
        plot energy ('E'), energy error ('dE'), or relative error energy ('de').
    xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.
	dpi  : int ,optional
        dots per inch
    plot_xaxis : bool, optional
        do you want to plot x-axis (y=0)?
    legend_label: str, optional
        label for legend
    legend_loc: int, optional
        location of legend
    legend_label2: str, optional
        label2 for legend
    tight_layout: bool, optional
        tight_layout figure

    returns
    -------
    none
        none if savefig not given.
    figure : file
        a file containing saving figure if savefig given.


    |
    """
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = np.array(data[0])/period
    pos = data[1]
    vel = data[2]
    pot = data[3]
    KE =  np.array([0.5*vi.dot(vi) for vi in vel])
    PE =  pot
    Et = KE + PE
    E0 = KE[0] + PE[0]
    dE = Et-E0
    Error = dE/E0

    if otherfile !=' ':
        #open data
        with open(otherfile,'rb') as file:
            data = pickle.load(file)
        #extract data
        time2 = np.array(data[0])/period
        pos2 = data[1]
        vel2 = data[2]
        pot2 = data[3]
        KE2 =  np.array([0.5*vi.dot(vi) for vi in vel2])
        PE2 =  pot2
        Et2 = KE2 + PE2
        E02 = KE2[0] + PE2[0]
        dE2 = Et2-E02
        Error2 = dE2/E02


    if type =='de':
        plt.plot(time, Error,label=legend_label)
        if otherfile !=' ':
            plt.plot(time2,Error2,label=legend_label2)
        plt.ylabel(r'$\Delta E / E_0$', fontsize=16)
    elif type=='dE':
        plt.plot(time,dE,label=legend_label)
        if otherfile !=' ':
            plt.plot(time2,dE2,label=legend_label2)
        plt.ylabel(r'$\Delta E$', fontsize=16)
    else:
        plt.plot(time,Et,label=legend_label)
        if otherfile !=' ':
            plt.plot(time2,Et2,label=legend_label2)
        plt.ylabel(r'$E$', fontsize=16)

    if period !=1.:
        plt.xlabel(r'$t/T$',fontsize=16)
    else:
        plt.xlabel(r'$t$',fontsize=16)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if plot_xaxis:
        plt.hlines(0,xlim[0],xlim[1],color='k')
    #add legend
    if legend_loc!=-1:
        plt.legend(loc=legend_loc)
    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    #tight_layout
    if tight_layout:
        plt.tight_layout()

    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)
    plt.show()

def plot_Lz_time(fname,type='dl',period=1., xlim=[0,0],ylim=[0,0], savefig=' ', dpi=300):
    '''
        plot angular momentum in Z-direction ('L') :math:`L_z` or error ('dL')
		:math:`\Delta L_z`,
	    relative Lz error ('dl') :math:`\Delta L_z / L_{z0}` over time.

    parameters
    ----------
    fname : str
        name of file containing t,x,v.
    type : str
        plot Lz ('L'), Lz error ('dL'), or relative error Lz ('dl').
    xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.
    dpi  : int, optional
        dots per inch

    returns
    -------
    none
        none if savefig not given.
    figure : file
        a file containing saving figure if savefig given.
	'''

    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = np.array(data[0])/period
    pos = data[1]
    vel = data[2]

    x = np.array([xt[0] for xt in pos])
    y = np.array([xt[1] for xt in pos])

    vx = np.array([vt[0] for vt in vel])
    vy = np.array([vt[1] for vt in vel])

    Lz = x*vy - y*vx
    Lz0 = Lz[0]
    dLz = Lz-Lz0
    Error = dLz/Lz0

    if type =='dl':
        plt.plot(time, Error)
        plt.ylabel(r'$\Delta L_z / L_{z0}$', fontsize=16)
    elif type=='dL':
        plt.plot(time,dLz)
        plt.ylabel(r'$\Delta L_z$', fontsize=16)
    else:
        plt.plot(time,Lz)
        plt.ylabel(r'$L_z$', fontsize=16)

    if period !=1.:
        plt.xlabel(r'$t/T$',fontsize=16)
    else:
        plt.xlabel(r'$t$',fontsize=16)

    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    plt.tight_layout()
    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)
    plt.show()

def info(fname):
    '''
        show command
    '''
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract RunInfoStr at last data
    RunInfo = data[len(data)-1]
    print(RunInfo)

def readsnapshot(fname):
    '''
        read snapshot file
    '''
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract Data snapshot
    snapsh = data[0]
    return snapsh

def makesnapfile(snp,fname):
    '''
       object of class snapshot + info -> file fname
    '''
    Info = "makesnapfile(snap = "+str(snp)+",fname = "+str(fname)+")"
    print (Info)
    dat = [snp, Info]
    with open(fname,'wb') as file:
        pickle.dump(dat,file)

def scatter_XY_snapshots(fnames,xlim=[0,0],ylim=[0,0],savefig=' ',dpi=300):
    #Name:
    #    scatter_XY_snapshot
    """simply plot orbit  XY

    parameters
    ----------
    fnames : list of str
        filename contains snapshot
    xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.

    returns
    -------
    none
        none if savefig not given.
    figure: file
        show figure and save figure if savefig given.

    |
    """

    #History
    # 03-11-2021  written Hasanuddin

    #open data, Data contains snapshot
    Data = []
    for fname in fnames:
        Data.append(readsnapshot(fname))
    # Extract Data,
    # Orbs list contains orbits
    Orbs = [ snp.orbits for snp in Data]
	# Torbs
    Torbs = [snp.time for snp in Data]
    # how many orbit in each snapshot
    Norb = len(Orbs[0])
	#Extract Orbits
    O    = []
    TO   = []
    for orb in Orbs:
        O = O + orb
    for t in Torbs:
        TO = TO + Norb*[t]
    #print(TO)
    Pos = [ o.x for o in O]
    #print (pos)
    #vel = data[2]

    X = [x[0] for x in Pos]
    Y = [x[1] for x in Pos]

    plt.scatter(X, Y,c=TO,cmap='hot',edgecolor='k')
    plt.colorbar()
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y$',fontsize=16)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    plt.axes().set_aspect('equal')
    plt.tight_layout()
    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)
    plt.show()

def calculate_area(fnames):
    """simply area phase space

    parameters
    ----------
    fnames : list of str
        filename contains snapshot

    """
    areas = []
    for i in range(len(fnames)):
        snap = readsnapshot(fnames[i])
        x10 = snap.orbits[1].x[0] - snap.orbits[0].x[0]
        x30 = snap.orbits[3].x[0] - snap.orbits[0].x[0]
        v10 = snap.orbits[1].v[0] - snap.orbits[0].v[0]
        v30 = snap.orbits[3].v[0] - snap.orbits[0].v[0]
        area = x10*v30 - x30*v10
        areas.append(area)
    print (areas)

def scatter_XVx_snapshots(fnames,xlim=[0,0],ylim=[0,0],savefig=' ',dpi=300):
    #Name:
    #    scatter_XVx_snapshot
    """simply scatter phase space

    parameters
    ----------
    fnames : list of str
        filename contains snapshot
    xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.
    dpi    : int, optional
    returns
    -------
    none
        none if savefig not given.
    figure: file
        show figure and save figure if savefig given.

    |
    """

    #History
    # 03-11-2021  written Hasanuddin

    #open data, Data contains snapshot
    Data = []
    for fname in fnames:
        Data.append(readsnapshot(fname))
    # Extract Data,
    # Orbs list contains orbits
    Orbs = [ snp.orbits for snp in Data]
	# Torbs
    Torbs = [snp.time for snp in Data]
    # how many orbit in each snapshot
    Norb = len(Orbs[0])
	#Extract Orbits
    O    = []
    TO   = []
    for orb in Orbs:
        O = O + orb
    for t in Torbs:
        TO = TO + Norb*[t]
    #print(TO)
    Pos = [ o.x for o in O]
    Vel = [ o.v for o in O]

    X = [x[0] for x in Pos]
    Vx =[v[0] for v in Vel]

    plt.scatter(X, Vx,c=TO,cmap='hot',edgecolor='k')
    #if Norb>2:
    #    plt.plot(X[1:],Vx[1:])
    plt.colorbar()
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$v_x$',fontsize=16)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    plt.axes().set_aspect('equal')
    plt.tight_layout()
    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)
    plt.show()
