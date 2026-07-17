#ifndef __PID_H
#define __PID_H

typedef struct{
	float Kp;
	float Ki;
	float Kd;
	float Error;
	float Last_Error;
	float Integral;
}PID_TypeDef;
float PID_Compute(PID_TypeDef *pid, float measure);
void Track_Control_Loop();

#endif
