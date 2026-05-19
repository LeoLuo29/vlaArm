
import time
import gpio_init
import drivers.SG92R
import drivers.MG995





## welcomming message
print("welcome to vlaArm mainloop")
graber = drivers.MG995.Mod_MG995(18)

##mainloop
cntLoop = 0
while True:
    print("________loop" + str(cntLoop) + "_______")
    graber.onRun()
    cntLoop += 1