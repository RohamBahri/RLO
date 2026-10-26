"""Local core-unit tests also work when the full RLO checkout isn't mounted."""
import importlib.util
import sys
import types
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
if importlib.util.find_spec('rlo.wave2_replay') is None:
    import numpy as np
    module=types.ModuleType('rlo.wave2_replay')
    class Snapshot:
        def __init__(self,instance,visited,current,load):
            self.instance,self.visited,self.current,self.load=(instance,visited,current,load)
        @classmethod
        def take(cls,env):
            return cls(env.instance,env.visited.copy(),env.current,env.load)
        def restore(self):
            return TinyEnv(self.instance,self.visited.copy(),self.current,self.load)
    class TinyEnv:
        def __init__(self,inst,visited=None,current=0,load=0):
            self.instance=inst;self.current=current;self.load=load
            self.visited=np.array([True,False,False]) if visited is None else visited
            self.dist=np.array([[0.,1.,2.],[1.,0.,1.],[2.,1.,0.]])
            self.tour=[self.current];self.done=bool(all(self.visited) and self.current==0)
            self.cost=0.
        def legal(self):
            return np.array([self.current!=0,not self.visited[1],not self.visited[2]])
        def observe(self):
            node=np.zeros((3,6),dtype=np.float32)
            node[:,0]=[0.,1.,2.];node[:,3]=[1,0,0]
            node[:,4]=self.visited;node[self.current,5]=1
            return node,np.array([0.,.5],np.float32),self.dist.astype('float32'),self.legal()
        def step(self,action):
            before=self.cost
            self.cost+=self.dist[self.current,action];self.current=action
            if action!=0:self.visited[action]=True
            self.tour.append(action)
            if self.visited.all() and self.current!=0:
                self.cost+=self.dist[self.current,0];self.current=0;self.tour.append(0)
            self.done=bool(self.visited.all() and self.current==0)
            return -(self.cost-before),self.done
    module.Snapshot=Snapshot
    sys.modules['rlo.wave2_replay']=module
    sys.modules['tiny_test_env']=types.SimpleNamespace(TinyEnv=TinyEnv)
if importlib.util.find_spec('rlo.env') is None:
    from tiny_test_env import TinyEnv
    m=types.ModuleType('rlo.env');m.CVRPEnv=TinyEnv
    sys.modules['rlo.env']=m
if importlib.util.find_spec('rlo.problem') is None:
    m=types.ModuleType('rlo.problem');m.validate=lambda inst,tour: 0.
    sys.modules['rlo.problem']=m
