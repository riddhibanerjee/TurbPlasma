#!/usr/bin/env python
import numpy as np
from commands import getstatusoutput as syscomout
from os.path import basename, realpath, exists
import AnalysisFunctions as af
from scipy.ndimage import gaussian_filter as gf

import sys
sys.path.append('/nethome/riddhib/Turbplasma')
  
class spc(object):
   """ SPectral Code Reader: First Cut
	First Cut - 2017/01/13
   """

   def __init__(self,shelldirname=None):
      # If no rundir specified
      if shelldirname is None: 
         shelldirname = raw_input('Please enter the rundir: ') 
      self.rundir = realpath(shelldirname)
      self.dirname= basename(self.rundir)
      self.primitives=['ax','ay','az','vx','vy','vz']
      self.derived={'bx':['ay','az'],'by':['ax','az'],'bz':['ax','ay'],\
        'ex':['by','bz','vy','vz'],'ey':['bx','bz','vx','vz'],\
        'ez':['bx','by','vx','vy']}
      self.allvars=self.primitives+['bx','by','bz','ex','ey','ez']
      self._readinit()

   def _readinit(self):
      f=open(self.rundir+'/mhd3-00-000.dat','rb')
      dum0=np.fromfile(f,dtype='int32',count=1)
      ih=np.fromfile(f,dtype='i4',count=9)
      f.close()
      self.nxc=ih[0]+1;self.nyc=ih[1];self.nzp=ih[2]
      self.nx=ih[3]; self.ny=ih[4]; self.nz=ih[5]
      self.lbox=float(2.0*np.pi*ih[6]); self.nprocs=ih[8]
      self._readglobs()
      self._readlengs()
      self._readspectra()
      self.xx=np.linspace(0,self.lbox,self.nx)
      self.yy=np.linspace(0,self.lbox,self.ny)
      self.zz=np.linspace(0,self.lbox,self.nz)
      self.dx=self.lbox/float(self.nx)
      self.dy=self.lbox/float(self.ny)
      self.dz=self.lbox/float(self.nz)
      #self.kmax=int(float(self.nx)/3.0) # Assuming nx=ny=nz
      self.kdiss1= ((self.enst + self.jsqd)**0.25)/np.sqrt(self.visc) # Kinetic dissipation wavenumber
      self.kdiss2= ((self.enst + self.jsqd)**0.25)/np.sqrt(self.rsist) # Magnetic dissipation wavenumber
      self.eps=self.visc*self.enst + self.rsist*self.jsqd
      self.urms=np.sqrt((2.0*self.Ev)/3.0)
      self.brms=np.sqrt((2.0*self.Eb)/3.0)
      self.zprms=np.sqrt(2.0*(self.Eb+self.Ev+self.Hc)/3.0)
      self.zmrms=np.sqrt(2.0*(self.Eb+self.Ev-self.Hc)/3.0)
      self.Lv_taylor=self.urms*(np.sqrt(15.0*self.visc/self.eps))
      self.Lb_taylor=self.brms*(np.sqrt(15.0*self.rsist/self.eps)) 
      self.Re_taylor=self.urms*(self.Lv_taylor/self.visc)
      self.Lv_int=(3.0*np.pi*self.Lv)/(4.0*self.Ev)
      self.Lb_int=(3.0*np.pi*self.Lb)/(4.0*self.Eb)
      self.Re_int=self.urms*(self.Lv_int/self.visc)
      self.Cv_eps=(self.eps*self.Lv_int)/(self.urms**3.0)
      self.Cb_eps=(self.eps*self.Lb_int)/(self.brms**3.0)
      self.C_eps=(self.eps*self.Lv_int)/(self.zprms**3.0)
      self.Re_minus=self.zprms*(self.Lv_int/(self.visc+self.rsist))


   def _readlengs(self):
      
      fl=open(self.rundir+'/spectra-lengths.dat','r')
      lengs = []
      
      for i in fl:
	 lengs += [i.split()]

      # lengs[0] is label, 
      # lengs[1] = [nflds, nwrt,  B0, nx, ny, nz, visc, rsist, dt, Hall_p, 0.0]
      # lengs[2] = [time, Ev, Eb, Hc, enst, jsqd, Lv, Lb, Lhc, 0.0]

      self.label0 = lengs[0]

      n = int(len(lengs)-2)
      
      self.Lt = np.zeros(n) 	# Time
      self.Lv = np.zeros(n)
      self.Lb = np.zeros(n)
      self.Lhc = np.zeros(n)

      for i in range(n):

         self.Lt[i] = lengs[2+i][0]
         self.Lv[i] = lengs[2+i][6]
         self.Lb[i] = lengs[2+i][7]
         self.Lhc[i] = lengs[2+i][8]

   def _readspectra(self):

      fl=open(self.rundir+'/spectra.dat','r')
      spectra = []
      
      for i in fl:
	 spectra += [i.split()]
	 
      # spectra[0] = label,
      # spectra[1] = k, Ev(k) ... 
      # spectra[2] = [nwrt, ktop(kmax), B0, Nx, Ny, Nz, visc, rsis, dt, Hall_p, 0.0]
      # spectra[3] = time
      # spectra[4] = [k, Ev(k), Eb(k), Hc(k), Hm(k), Hk(k), mult]

      self.label1 = spectra[0]
      self.knwrt = int(spectra[2][0]); self.kmax = int(spectra[2][1])
      
      n = int(self.knwrt)
      #n = 2

      self.kt = np.zeros(self.knwrt) 	# Time for a particluar k-E(k) value
      self.kk = np.zeros(self.kmax)
      self.Evk = np.zeros((self.kmax,self.knwrt))
      self.Ebk = np.zeros((self.kmax,self.knwrt))
      self.Hck = np.zeros((self.kmax,self.knwrt))
      self.Hmk = np.zeros((self.kmax,self.knwrt))
      self.Hkk = np.zeros((self.kmax,self.knwrt))

      ## There are kmax number of k's but the k's go like 0,1,2 ... kmax. 
      ## So there are kmax+1 number of k's with the 0th mode meaning nothing

      for i in range(n):
	 #print  spectra[3+i*(self.kmax+2)]
	 
	 for j in range(self.kmax):
		 
	         self.kk[j] = spectra[5+j][0]		# kk=0 is nothing
		 
		 self.Evk[j][i] = spectra[5+i*(self.kmax+2)+j][1]
		 self.Ebk[j][i] = spectra[5+i*(self.kmax+2)+j][2]
		 self.Hck[j][i] = spectra[5+i*(self.kmax+2)+j][3]
		 self.Hmk[j][i] = spectra[5+i*(self.kmax+2)+j][4]
		 self.Hkk[j][i] = spectra[5+i*(self.kmax+2)+j][5]


   def _readglobs(self):
      
      # Riddhi's code
      fl=open(self.rundir+'/globs.dat','r')

      globs = []

      for i in fl:
     	 globs += [i.split()]

      # globs[0] is label, 
      # globs[1] = [nwrt,  B0, nx, ny, nz, visc, rsist, dt, Hall_p, 0.0]
      # globs[2] = [time, Ev, Eb, asqd, jsqd]
      # globs[3] = [enst, Hm, Hc, Hk, 0.0]
      # globs[4] = [vxb1, vxb2, vxb3]
      # globs[5] = [Ev_2D, Eb_2D, asqd_2D, jsqd_2D, Enst_2D]
      # globs[6] = [Hc_2D]


      self.label = globs[0]
      self.nwrt = globs[1][0]; self.B0 = globs[1][1]; self.nx = globs[1][2]; self.ny = globs[1][3]; self.nz = globs[1][4]; self.visc = float(globs[1][5]); self.rsist = float(globs[1][6]); self.dt = globs[1][7]; self.Hall_p = globs[1][8]

      n = int(((len(globs)-6)/5.0) + 1.0)

      self.time  = np.zeros(n)
      self.Ev    = np.zeros(n)
      self.Eb    = np.zeros(n)
      self.asqd  = np.zeros(n)
      self.jsqd  = np.zeros(n)
      self.enst  = np.zeros(n)
      self.Hm    = np.zeros(n)
      self.Hc    = np.zeros(n)
      self.Hk    = np.zeros(n)
      self.VcBx  = np.zeros(n)
      self.VcBy  = np.zeros(n)
      self.VcBz  = np.zeros(n)
      self.Ev_2D = np.zeros(n)
      self.Eb_2D = np.zeros(n)
      self.asqd_2D = np.zeros(n)
      self.jsqd_2D = np.zeros(n)
      self.enst_2D = np.zeros(n)
      self.Hc_2D = np.zeros(n)

    
      for i in range(n):

         self.time[i] = globs[2+i*5][0]
         self.Ev[i] = globs[2+i*5][1]
         self.Eb[i] = globs[2+i*5][2]
         self.asqd[i] = globs[2+i*5][3]
	 self.jsqd[i] = globs[2+i*5][4]

	 self.enst[i] = globs[3+i*5][0]
	 self.Hm[i] = globs[3+i*5][1]
	 self.Hc[i] = globs[3+i*5][2]
	 self.Hk[i] = globs[3+i*5][3]

	 self.VcBx[i] = globs[4+i*5][0]
         self.VcBy[i] = globs[4+i*5][1]
         self.VcBz[i] = globs[4+i*5][2]

         self.Ev_2D[i] = globs[5+i*5][0]
         self.Eb_2D[i] = globs[5+i*5][0]
         self.asqd_2D[i] = globs[5+i*5][0]
         self.jsqd_2D[i] = globs[5+i*5][0]
         self.enst_2D[i] = globs[5+i*5][0]

	 self.Hc_2D[i] = globs[6+i*5][0]
      

   def vars2load(self,v2lu):
      """
         Based on user input, find the dependencies and create the v2l
         array.
      """
      
      if len(v2lu) == 1:
         if v2lu[0] == 'min':
               self.vars2l=self.primitives
         else:
               self.vars2l=v2lu
      else:
         self.vars2l = v2lu

      v2l=v2lu[:]
      
      while any([x in self.derived for x in v2l]):
         toload=v2l[:]
         v2l=[]
         while len(toload) > 0:
            current=toload.pop()
            if current in self.primitives and current not in v2l:
              v2l.append(current)
            elif current in self.derived:
              for i in self.derived[current]:
                if i not in v2l:
                  v2l.append(i)
            elif current not in self.primitives+self.derived.keys():
              print current+' not implemented. Implement it!'
      for i in v2lu:
        if i in self.derived: 
          for j in self.derived[i]:
            if j not in v2l: v2l.append(j)
      for i in v2lu:
        if i in self.derived and i not in v2l: v2l.append(i)
      self.vars2l=v2l
#     self.vars2l=sorted(v2l,key=self.allvars.index)
      
      for i in self.primitives:
         self.__dict__[i+'c']=np.zeros((self.nxc,self.ny,self.nz),dtype='complex')
         self.__dict__[i]    =np.zeros((self.nx, self.ny, self.nz))
      for i in self.vars2l:
         if i in self.derived:
            self.__dict__[i]=np.zeros((self.nx,self.ny,self.nz))

   def loadslice(self,timeslice):
#     from scipy.io import FortranFile
      for i in range(self.nprocs):
         f=open('mhd3-%02d-%03d.dat'%(timeslice,i),'rb')      
         d={}
         dum0=np.fromfile(f,dtype='int32',count=1)
         ih=np.fromfile(f,dtype='i4',count=9)
         dum1=np.fromfile(f,dtype='int32',count=1)
         ncount=(ih[0]+1)*ih[1]*ih[2]
         
         dum2=np.fromfile(f,dtype='int32',count=1)
         d['time']=np.fromfile(f,dtype='float',count=1)
         d['ax']  =np.fromfile(f,dtype='complex',count=ncount).reshape((ih[0]+1,ih[1],ih[2]),order='F')
         d['ay']  =np.fromfile(f,dtype='complex',count=ncount).reshape((ih[0]+1,ih[1],ih[2]),order='F')
         d['az']  =np.fromfile(f,dtype='complex',count=ncount).reshape((ih[0]+1,ih[1],ih[2]),order='F')
         dum2=np.fromfile(f,dtype='int32',count=1)
         
         dum2=np.fromfile(f,dtype='int32',count=1)
         d['time']=np.fromfile(f,dtype='float',count=1)
         d['vx']  =np.fromfile(f,dtype='complex',count=ncount).reshape((ih[0]+1,ih[1],ih[2]),order='F')
         d['vy']  =np.fromfile(f,dtype='complex',count=ncount).reshape((ih[0]+1,ih[1],ih[2]),order='F')
         d['vz']  =np.fromfile(f,dtype='complex',count=ncount).reshape((ih[0]+1,ih[1],ih[2]),order='F')
         dum2=np.fromfile(f,dtype='int32',count=1)
         f.close()

         for j in self.primitives:
            self.__dict__[j+'c'][:,:,i*ih[2]:(i+1)*ih[2]]=d[j]
      print 'Loaded primitives in Fourier Space'

      for i in self.primitives:
         self.__dict__[i] = np.fft.irfftn(self.__dict__[i+'c'].T)
      print 'Transformed primitives to real space'
      
      for i in self.vars2l:
         if i in self.derived:
            self.__dict__[i] = self._derivedv(i)
            

   def _derivedv(self,varname):
      import AnalysisFunctions as af
      if varname == 'bx':
         return af.pcurlx(self.ay,self.az)
      if varname == 'by':
         return af.pcurly(self.az,self.ax)
      if varname == 'bz':
         return af.pcurlz(self.ax,self.ay)
      if varname == 'ex':
         return self.vy*self.bz - self.vz*self.by
      if varname == 'ey':
         return self.vz*self.bx - self.vx*self.bz
      if varname == 'ez':
         return self.vx*self.by - self.vy*self.bx 
