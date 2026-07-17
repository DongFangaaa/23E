#include "stm32f10x.h"  
#include "Track.h"
#include "motor.h"
#include "Delay.h"

extern float last_error;

typedef struct{
	float Kp;
	float Ki;
	float Kd;
	float Error;
	float Last_Error;
	float Integral;
}PID_TypeDef;

float PID_Compute(PID_TypeDef *pid, float measure)
{
	pid->Error = measure;//更新
	
	pid->Integral += pid->Error;
	
	float output =(pid->Kp * pid->Error) + 
                (pid->Ki * pid->Integral) + 
                (pid->Kd * (pid->Error - pid->Last_Error));
	
	pid->Last_Error = pid->Error;
	
	return output;
}

PID_TypeDef Track_PID = {15.0f, 0.0f, 5.0f, 0, 0, 0}; // 初始参数：大P小D，无I

int ApplyDeadzone(int speed)
{
    if(speed == 0)    return 0;
    if(speed > 0)     return speed < 35 ? 35 : speed;  // 正转最低35
    if(speed < 0)     return speed > -35 ? -35 : speed; // 反转最低-35
    return 0;
}

void Track_Control_Loop() 
{
	uint16_t Black_Count = 0;
	uint8_t status = Track_State();
	float current_error = Track_Error();//存入偏移
	
	static uint8_t last_status = 0;

	int left_speed = 0;
  int right_speed = 0;
	
	if(current_error == 10)
	{
		Black_Count++;//记录全黑次数
	}
    // 丢线时强制转向，不走PID
    if(status == 0x1F)
    {
        if(last_error > 2.0f)
        {
            left_speed  = 70;
            right_speed = -70;
        }
        else if(last_error < -2.0f)
        {
            left_speed  = -70;
            right_speed =  70;
        }
        else
        {
            left_speed  = 35;
            right_speed = 35;
        }
    }
    // 1. 获取当前传感器偏差 (-5 到 5)
    
	else
	{
		if(last_status == 0x1F)
        {
            Track_PID.Last_Error = current_error;  // 重置Last_Error
            Track_PID.Integral = 0;                // 顺便清积分
        }
				
    int Base_Speed = (current_error >= 2.0f || current_error <= -2.0f) ?  35: 40;
	
    // 2. 计算 PID 调整值 
    float adjust = PID_Compute(&Track_PID, (float)current_error);
    
    // 3. 左右轮速度分配 (左加右减实现转向)
    left_speed  = Base_Speed + (int)adjust;
    right_speed = Base_Speed - (int)adjust;
				
		left_speed  = ApplyDeadzone(left_speed);
    right_speed = ApplyDeadzone(right_speed);
	
		if(left_speed <= -100)
			left_speed = -100;
		if(left_speed >= 100)
			left_speed = 100;
		if(right_speed <= -100)
			right_speed = -100;
    if(right_speed >= 100)
			right_speed = 100;
    // 4. 调用你编译成功的底层函数输出 PWM
		
		if(current_error <= -4.0f)
		{
			left_speed = -50;
			right_speed = 80;
		}
		if(current_error >= 4.0f &&current_error <= 5.0f)
		{
			left_speed = 80;
			right_speed = -50;
		}
	}
		if(Black_Count >= 4)
		{
			left_speed = 0;
			left_speed = 0;
		}
		if(Black_Count == 2)
		{
			left_speed = -50;
			left_speed = 80;
		}				
		if(Black_Count == 3)
		{
			left_speed = 80;
			left_speed = -50;
		}
		last_status = status;  // 更新上一次状态	
    Mptor_SetSpeed(left_speed, right_speed); 
		if(Black_Count >= 4)
		{
			Delay_s(2);
		}
}
