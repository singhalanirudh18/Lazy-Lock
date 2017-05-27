import RPi.GPIO as GPIO  
import time
GPIO.setwarnings(False)# this lets us have a time delay (see line 15)  
GPIO.setmode(GPIO.BCM)     # set up BCM GPIO numbering  
GPIO.setup(8, GPIO.IN)    # set GPIO25 as input (button)  
GPIO.setup(24, GPIO.OUT)   # set GPIO24 as an output (LED)  import time
knocks=0
#xylo_op
i=0
j=0
detect=0
#timer
gap=[1000,1000,1000,0]

def knock{} :
    while True:            # this will carry on until you hit CTRL+C
        if detect==0:
            if GPIO.input(8):
                knocks=knocks+1
                print(knocks)
                time.sleep((gap[i]-0.25*gap[i])/1000)
                i=i+1
                detect=1
        else:
            timer=time.time()*1000
            while time.time() < (timer+0.5*gap[i])/1000 :
                #xylo_op=GPIO.input(3)
                if GPIO.input(8):
                    knocks=knocks+1
                    print(knocks)
                    time.sleep((gap[i]-0.25*gap[i])/1000)
                    i=i+1
            if knocks==3 :
                detect=0
                j=1
                GPIO.output(24, 1)
                print("ho gaya")
                break
    

    

try:  
    knock()
    
  
finally:                   # this block will run no matter how the try block exits  
    GPIO.cleanup()         # clean up after yourself  



"""
knocks=0
#xylo_op
i=0
detect=0
#timer
gap=[1000,1000,1000,0]

while True:
    if detect==0:
        xylo_op=GPIO.input(3)
        if xylo_op==1:
            knocks=knocks+1
            time.sleep((gap[i]-0.25*gap[i])/1000)
            i=i+1
            detect=1
    else:
        timer=time.time()*1000
        while time.time() < (timer+0.5*gap[i])/1000 :
            xylo_op=GPIO.input(3)
            if xylo_op==1:
                knocks=knocks+1
                time.sleep((gap[i]-0.25*gap[i])/1000)
                i=i+1
        if knocks==3 :
            detect=0
            GPIO.output(5, 1)
            print("ho gaya")
                



while True :
    xylo_op=GPIO.input(4)

    print(xylo_op)
    GPIO.cleanup()
    
    if xylo_op==1:
        print(i)
        i+=1
    
    """

        
             
