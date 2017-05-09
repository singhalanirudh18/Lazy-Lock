
int knocks=0;
int gap[3];
int xylo_op;
unsigned long timer;
int i=0 , detect=0;
gap[0]=1000;
gap[1]=1000;
gap[2]=0;

void setup() {
  // put your setup code here, to run once:

}

void loop() {
  xylo_op = digitalRead(xylo);
  if(detect == 0){
    if(xylo_op == HIGH){
      knocks++;
      delay(gap[i]-0.25*gap[i]);
      i++;
      detect=1;
    }
  }
  if(detect == 1){
    timer = millis();
    while(millis() < (timer + 0.5*gap[i])){
      xylo_op = digitalRead(xylo);
      if(xylo_op == HIGH){
      knocks++;
      delay(gap[i]-0.25*gap[i]);
      i++;
      }      
      if( knocks == 3 ){
        detect=0;
        digitalWrite(door, HIGH)
      }
  
    }
    
  }
}
