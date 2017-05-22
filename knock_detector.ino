int knocks=0;
int gap[3];
int xylo_op;
unsigned long timer;
int i=0 , detect=0;
int xylo=2;
int door=8;

void setup() {
gap[0]=1000;
gap[1]=1000;
gap[2]=1000;
  pinMode(xylo,INPUT);
pinMode(door,OUTPUT);
Serial.begin(9600);
  // put your setup code here, to run once:

}

void loop() {
  if(detect == 0){
    
  xylo_op = digitalRead(xylo);
    if(xylo_op == HIGH){
      knocks++;
      delay(gap[i]-0.25*gap[i]);
      i++;
      detect=1;
    // digitalWrite(door, HIGH);
     //delay(500);
     Serial.println (knocks);
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
       Serial.println (knocks);
       Serial.println (timer);
      }      
      if( knocks == 3 ){
        detect=0;
        digitalWrite(door, HIGH);
        break;
        
      }    
  
}
 }
}
