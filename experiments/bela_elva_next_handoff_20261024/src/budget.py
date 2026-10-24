"""Offline budget validation, not model integration."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Plan:
    name: str
    stages: tuple
    def validate(self):
        if sum(l for l,k in self.stages)!=32:
            raise ValueError("require 32 decoder layers")
        if any(type(l)!=int or type(k)!=int or l<=0 or k<=0 for l,k in self.stages):
            raise ValueError("require positive integer layers and tokens")
        ks=[k for l,k in self.stages]
        if ks!=sorted(ks,reverse=True):
            raise ValueError("cannot restore pruned tokens")
        return self
    @property
    def W(self):
        self.validate()
        return sum(l*k for l,k in self.stages)
    @property
    def Q(self):
        self.validate()
        return sum(l*k*k for l,k in self.stages)

def compatible_deep(W, k0, kmid, layers=(4,8,20)):
    if any(type(x)!=int for x in (W,k0,kmid,*layers)):
        raise ValueError("integer inputs required")
    n=W-layers[0]*k0-layers[1]*kmid
    if n%layers[2]: return None
    kd=n//layers[2]
    return kd if 1<=kd<=kmid<=k0 else None

ARMS={
    "A0":Plan("A0",((4,172),(8,90),(20,32))),
    "A2":Plan("A2",((4,172),(8,120),(20,20))),
    "A3":Plan("A3",((4,172),(8,130),(20,16))),
    "B1":Plan("B1",((4,172),(8,90),(20,20))),
    "B2":Plan("B2",((4,172),(8,120),(20,32))),
    "W1024_probe":Plan("W1024_probe",((4,96),(8,50),(20,12))),
    "W4096_probe":Plan("W4096_probe",((4,256),(8,184),(20,80))),
    "W1024_historical":Plan("W1024_historical",((10,60),(10,28),(12,12))),
}
if __name__=="__main__":
    for name,p in ARMS.items():
        print(f"{name:20} W={p.W:5} avgK={p.W/32:7.2f} Q={p.Q}")
