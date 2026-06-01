
import time
import gpio_init
import drivers.SG92R
import drivers.MG995
import drivers.RDX3225





## welcomming message
print("welcome to vlaArm mainloop")
arm = drivers.RDX3225.Mod_RDX3225(18)

##mainloop
cntLoop = 0
while True:
    print("________loop" + str(cntLoop) + "_______")
    arm.onRun()
    cntLoop += 1