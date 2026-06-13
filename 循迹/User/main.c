#include "stm32f10x.h"                  // Device header
#include "OLED.h"
#include "Delay.h"
#include "Motor.h"
#include "Buzzer.h"
#include "PWM.h"
#include "HCSR04.h"


extern uint32_t Timer_Tick;

int main()
{
	Buzzer_Init();
	Motor_Init();
	Track_Init();
	HCSR04_Init();
	Timer_Init();
	
	uint32_t Last_Tick = 0;
	float Distance = 0;
	
	while (1)
	{
		if(Last_Tick != Timer_Tick)
		{	
			Track_Control_Loop();
			Last_Tick = Timer_Tick;
			if(Last_Tick % 100 == 0)
			{
				Distance = HCSR04_GetValue();
			}
			if(Distance <= 30 && Distance != 0)
			{
				Buzzer_Beep();
			}
			Buzzer_Tick_Task();
		}
	}
}
