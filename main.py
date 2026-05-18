
import time
import gpio_init
import SG92R





## welcomming message
print("welcome to vlaArm mainloop")
graber = SG92R.Mod_SG92R()

##mainloop
cntLoop = 0
while True:
    print("________loop" + str(cntLoop) + "_______")
    graber.onRun(state=True)
    cntLoop += 1